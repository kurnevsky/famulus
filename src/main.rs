mod config;
mod infill;
mod llama_cpp;
mod mistral;
mod ollama;

use std::{collections::HashMap, env, fs::File, io::BufReader, sync::Arc};

use anyhow::{anyhow, Result};
use clap::Command;
use config::Config;
use infill::Infill;
use lsp_server::{Connection, ErrorCode, Message, RequestId, Response};
use lsp_types::{
  notification::{Cancel, DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Exit, Notification},
  request::{ExecuteCommand, InlineCompletionRequest, Request},
  CancelParams, DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
  ExecuteCommandOptions, ExecuteCommandParams, InlineCompletionItem, InlineCompletionParams, InlineCompletionResponse,
  NumberOrString, OneOf, Range, ServerCapabilities, TextDocumentSyncKind, Uri, WorkDoneProgressOptions,
};
use ropey::Rope;
use serde::Deserialize;
use tokio::task::JoinHandle;

#[derive(Debug)]
struct Document {
  rope: Rope,
  version: i32,
}

#[derive(Debug, Default)]
struct State {
  documents: HashMap<Uri, Document>,
  tasks: Arc<papaya::HashMap<RequestId, JoinHandle<Result<()>>>>,
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LlmRefactorParams {
  pub uri: Uri,
  pub range: Range,
}

const LLM_REFACTOR_COMMAND: &str = "llm-refactor";

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let _matches = Command::new(clap::crate_name!())
    .version(clap::crate_version!())
    .author(clap::crate_authors!("\n"))
    .about(clap::crate_description!())
    .get_matches();

  let (connection, io_threads) = Connection::stdio();
  let server_capabilities = ServerCapabilities {
    execute_command_provider: Some(ExecuteCommandOptions {
      commands: vec![LLM_REFACTOR_COMMAND.to_string()],
      work_done_progress_options: WorkDoneProgressOptions {
        work_done_progress: Some(false),
      },
    }),
    inline_completion_provider: Some(OneOf::Left(true)),
    text_document_sync: Some(lsp_types::TextDocumentSyncCapability::Kind(
      TextDocumentSyncKind::INCREMENTAL,
    )),
    ..Default::default()
  };
  let initialization_args = connection.initialize(serde_json::to_value(server_capabilities)?)?;

  let config = {
    #[derive(Clone, PartialEq, Debug, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct InitializationArgs {
      initialization_options: Config,
    }
    serde_json::from_value::<InitializationArgs>(initialization_args)?.initialization_options
  };
  let infill = config.get_infill();

  let sender = Arc::new(connection.sender);
  let client = Arc::new(reqwest::Client::new());

  let mut state = State::default();

  for msg in &connection.receiver {
    match msg {
      Message::Request(request) => {
        if request.method == InlineCompletionRequest::METHOD {
          let (request_id, params) = request.extract::<InlineCompletionParams>(InlineCompletionRequest::METHOD)?;
          let document = state
            .documents
            .get(&params.text_document_position.text_document.uri)
            .ok_or_else(|| {
              anyhow!(
                "Missing document: {}",
                params.text_document_position.text_document.uri.as_str()
              )
            })?;

          let index = document
            .rope
            .line_to_char(params.text_document_position.position.line as usize)
            + params.text_document_position.position.character as usize;
          let prefix = document.rope.slice(0..index).to_string();
          let suffix = document.rope.slice(index..document.rope.len_chars()).to_string();

          let mistral_infill = infill.clone();
          let client = client.clone();
          let sender = sender.clone();
          let tasks = state.tasks.clone();
          let request_id_c = request_id.clone();
          let future = async move {
            let completions = mistral_infill.infill(client, prefix, suffix).await;
            match completions {
              Result::Ok(completions) => {
                let range = Range::new(
                  params.text_document_position.position,
                  params.text_document_position.position,
                );
                let completion_items = completions
                  .filter(|completion| !completion.is_empty())
                  .map(|completion| InlineCompletionItem {
                    insert_text: completion,
                    filter_text: None,
                    range: Some(range),
                    command: None,
                    insert_text_format: None,
                  })
                  .collect();
                tasks.pin().remove(&request_id_c);
                sender.send(Message::Response(Response::new_ok(
                  request_id_c,
                  InlineCompletionResponse::Array(completion_items),
                )))?;
              }
              Result::Err(error) => {
                tasks.pin().remove(&request_id_c);
                sender.send(Message::Response(Response::new_err(
                  request_id_c,
                  ErrorCode::RequestFailed as i32,
                  format!("Failed to get response: {}", error),
                )))?;
              }
            }
            Ok(())
          };
          let handle = tokio::task::spawn(future);
          state.tasks.pin().insert(request_id, handle);
        } else if request.method == ExecuteCommand::METHOD {
          let (request_id, params) = request.extract::<ExecuteCommandParams>(ExecuteCommand::METHOD)?;
          let command_str = params.command.as_str();
          if command_str == LLM_REFACTOR_COMMAND {
          } else {
            sender.send(Message::Response(Response::new_err(
              request_id,
              ErrorCode::InvalidRequest as i32,
              format!("Unknown command: {}", command_str),
            )))?;
          }
        }
      }
      Message::Notification(notification) => {
        if notification.method == DidOpenTextDocument::METHOD {
          let params: DidOpenTextDocumentParams = serde_json::from_value(notification.params)?;
          let file = File::open(params.text_document.uri.path().as_str())?;
          let reader = BufReader::new(file);
          let rope = Rope::from_reader(reader)?;
          state.documents.insert(
            params.text_document.uri,
            Document {
              rope,
              version: params.text_document.version,
            },
          );
        } else if notification.method == DidCloseTextDocument::METHOD {
          let params: DidCloseTextDocumentParams = serde_json::from_value(notification.params)?;
          state.documents.remove(&params.text_document.uri);
        } else if notification.method == DidChangeTextDocument::METHOD {
          let params: DidChangeTextDocumentParams = serde_json::from_value(notification.params)?;
          for change in params.content_changes {
            if let Some(range) = change.range {
              let document = state
                .documents
                .get_mut(&params.text_document.uri)
                .ok_or_else(|| anyhow!("Missing document: {}", params.text_document.uri.as_str()))?;
              let start_index = document.rope.line_to_char(range.start.line as usize) + range.start.character as usize;
              let end_index = document.rope.line_to_char(range.end.line as usize) + range.end.character as usize;
              document.rope.remove(start_index..end_index);
              document.rope.insert(start_index, &change.text);
              document.version = params.text_document.version;
            } else {
              let rope = Rope::from_str(&change.text);
              state.documents.insert(
                params.text_document.uri.clone(),
                Document {
                  rope,
                  version: params.text_document.version,
                },
              );
            }
          }
        } else if notification.method == Cancel::METHOD {
          let params: CancelParams = serde_json::from_value(notification.params)?;
          let id: RequestId = match params.id {
            NumberOrString::Number(id) => id.into(),
            NumberOrString::String(id) => id.into(),
          };
          if let Some(handle) = state.tasks.pin().remove(&id) {
            handle.abort();
          }
        } else if notification.method == Exit::METHOD {
          return Ok(());
        }
      }
      _ => {}
    }
  }

  io_threads.join().map_err(|e| e.into())
}
