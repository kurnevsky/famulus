mod chat;
mod config;
mod infill;
mod llama_cpp;
mod mistral;
mod ollama;
mod openai;

use std::{collections::HashMap, env, fs::File, io::BufReader, sync::Arc};

use anyhow::{anyhow, Result};
use chat::Chat;
use clap::Command;
use config::Config;
use crossbeam_channel::Sender;
use dashmap::DashMap;
use derive_more::From;
use infill::Infill;
use lsp_server::{Connection, ErrorCode, Message, Request as LspRequest, RequestId, Response as LspResponse};
use lsp_types::{
  notification::{Cancel, DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Exit, Notification},
  request::{ApplyWorkspaceEdit, ExecuteCommand, InlineCompletionRequest, Request},
  ApplyWorkspaceEditParams, CancelParams, DidChangeTextDocumentParams, DidCloseTextDocumentParams,
  DidOpenTextDocumentParams, DocumentChanges, ExecuteCommandOptions, ExecuteCommandParams, InitializeParams,
  InlineCompletionItem, InlineCompletionParams, InlineCompletionResponse, Location, NumberOrString, OneOf,
  OptionalVersionedTextDocumentIdentifier, Range, ServerCapabilities, TextDocumentEdit, TextDocumentSyncKind, TextEdit,
  Uri, WorkDoneProgressOptions, WorkspaceEdit,
};
use ramhorns::{encoding::Encoder, Content, Template};
use reqwest::Client;
use ropey::{Rope, RopeSlice};
use serde_json::Value;
use tokio::task::JoinHandle;

#[derive(From)]
struct RopeSliceContent<'a>(RopeSlice<'a>);

impl Content for RopeSliceContent<'_> {
  #[inline]
  fn is_truthy(&self) -> bool {
    self.0.len_chars() > 0
  }

  #[inline]
  fn capacity_hint(&self, _tpl: &Template) -> usize {
    self.0.len_chars()
  }

  #[inline]
  fn render_escaped<E: Encoder>(&self, encoder: &mut E) -> Result<(), E::Error> {
    for chunk in self.0.chunks() {
      encoder.write_escaped(chunk)?;
    }
    Ok(())
  }

  #[inline]
  fn render_unescaped<E: Encoder>(&self, encoder: &mut E) -> Result<(), E::Error> {
    for chunk in self.0.chunks() {
      encoder.write_unescaped(chunk)?;
    }
    Ok(())
  }
}

#[derive(Content)]
struct SelectionContent<'a> {
  prompt: String,
  selection: RopeSliceContent<'a>,
  prefix: RopeSliceContent<'a>,
  suffix: RopeSliceContent<'a>,
}

#[derive(Debug)]
struct Document {
  rope: Rope,
  version: i32,
}

#[derive(Debug)]
struct State {
  document_changes: bool,
  sender: Arc<Sender<Message>>,
  client: Arc<Client>,
  config: Config,
  documents: Arc<DashMap<Uri, Document>>,
  tasks: Arc<DashMap<RequestId, JoinHandle<Result<()>>>>,
  templates: Vec<Template<'static>>,
}

impl State {
  fn inline_completion_request(&self, request_id: RequestId, params: InlineCompletionParams) -> Result<()> {
    let document = self
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
    let prefix = document.rope.slice(..index).to_string();
    let suffix = document.rope.slice(index..).to_string();

    let infill = self.config.get_infill();
    let client = self.client.clone();
    let sender = self.sender.clone();
    let tasks = self.tasks.clone();
    let request_id_c = request_id.clone();
    let future = async move {
      let completions = infill.infill(client, prefix, suffix).await;
      match completions {
        Result::Ok(completions) => {
          tasks.remove(&request_id_c);
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
          sender.send(Message::Response(LspResponse::new_ok(
            request_id_c,
            InlineCompletionResponse::Array(completion_items),
          )))?;
        }
        Result::Err(error) => {
          tasks.remove(&request_id_c);
          sender.send(Message::Response(LspResponse::new_err(
            request_id_c,
            ErrorCode::RequestFailed as i32,
            format!("Failed to get response: {}", error),
          )))?;
        }
      }
      Ok(())
    };
    let handle = tokio::task::spawn(future);
    self.tasks.insert(request_id, handle);
    Ok(())
  }

  fn rewrite(&self, request_id: RequestId, arguments: Vec<Value>) -> Result<()> {
    match TryInto::<[_; 2]>::try_into(arguments) {
      Ok([location, prompt]) => {
        let location: Location = serde_json::from_value(location)?;
        let prompt: String = serde_json::from_value(prompt)?;
        let document = self
          .documents
          .get(&location.uri)
          .ok_or_else(|| anyhow!("Missing document: {}", location.uri.as_str()))?;
        let start_index =
          document.rope.line_to_char(location.range.start.line as usize) + location.range.start.character as usize;
        let end_index =
          document.rope.line_to_char(location.range.end.line as usize) + location.range.end.character as usize;
        let content = SelectionContent {
          prompt,
          selection: document.rope.slice(start_index..end_index).into(),
          prefix: document.rope.slice(..start_index).into(),
          suffix: document.rope.slice(end_index..).into(),
        };
        let messages = self
          .config
          .chat
          .messages
          .iter()
          .zip(self.templates.iter())
          .map(|(message, template)| (message.role.clone(), template.render(&content)))
          .collect();
        let version = document.version;
        let chat = self.config.get_chat();
        let client = self.client.clone();
        let document_changes = self.document_changes;
        let sender = self.sender.clone();
        let tasks = self.tasks.clone();
        let documents = self.documents.clone();
        let request_id_c = request_id.clone();
        let future = async move {
          let choices = chat.chat(client.clone(), messages).await;
          match choices {
            Ok(mut choices) => {
              tasks.remove(&request_id_c);
              sender.send(Message::Response(LspResponse::new_ok(request_id_c, ())))?;
              if let Some(choice) = choices.next() {
                if !documents
                  .get(&location.uri)
                  .map_or(false, |document| document.version == version)
                {
                  return Ok(());
                }
                let edit_params = if document_changes {
                  ApplyWorkspaceEditParams {
                    label: None,
                    edit: WorkspaceEdit {
                      changes: None,
                      document_changes: Some(DocumentChanges::Edits(vec![TextDocumentEdit {
                        text_document: OptionalVersionedTextDocumentIdentifier {
                          uri: location.uri,
                          version: Some(version),
                        },
                        edits: vec![OneOf::Left(TextEdit {
                          range: location.range,
                          new_text: choice,
                        })],
                      }])),
                      change_annotations: None,
                    },
                  }
                } else {
                  ApplyWorkspaceEditParams {
                    label: None,
                    edit: WorkspaceEdit {
                      changes: Some(HashMap::from([(
                        location.uri,
                        vec![TextEdit {
                          range: location.range,
                          new_text: choice,
                        }],
                      )])),
                      document_changes: None,
                      change_annotations: None,
                    },
                  }
                };
                sender.send(Message::Request(LspRequest::new(
                  RequestId::from(0),
                  ApplyWorkspaceEdit::METHOD.to_string(),
                  edit_params,
                )))?;
              }
            }
            Err(error) => {
              tasks.remove(&request_id_c);
              sender.send(Message::Response(LspResponse::new_err(
                request_id_c,
                ErrorCode::RequestFailed as i32,
                format!("Failed to get response: {}", error),
              )))?;
            }
          }
          Ok(())
        };
        let handle = tokio::task::spawn(future);
        self.tasks.insert(request_id, handle);
        Ok(())
      }
      Err(arguments) => Err(anyhow!("Wrong number of arguments: {}", arguments.len())),
    }
  }

  fn did_open_text_document(&mut self, params: DidOpenTextDocumentParams) -> Result<()> {
    let file = File::open(params.text_document.uri.path().as_str())?;
    let reader = BufReader::new(file);
    let rope = Rope::from_reader(reader)?;
    self.documents.insert(
      params.text_document.uri,
      Document {
        rope,
        version: params.text_document.version,
      },
    );
    Ok(())
  }

  fn did_close_text_document(&mut self, params: DidCloseTextDocumentParams) {
    self.documents.remove(&params.text_document.uri);
  }

  fn did_change_text_document(&mut self, params: DidChangeTextDocumentParams) -> Result<()> {
    for change in params.content_changes {
      if let Some(range) = change.range {
        let mut document = self
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
        self.documents.insert(
          params.text_document.uri.clone(),
          Document {
            rope,
            version: params.text_document.version,
          },
        );
      }
    }
    Ok(())
  }

  fn cancel(&self, params: CancelParams) {
    let id: RequestId = match params.id {
      NumberOrString::Number(id) => id.into(),
      NumberOrString::String(id) => id.into(),
    };
    if let Some((_, handle)) = self.tasks.remove(&id) {
      handle.abort();
    }
  }
}

const REWRITE_COMMAND: &str = "famulus-rewrite";

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
      commands: vec![REWRITE_COMMAND.to_string()],
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
  let initialize_params = connection.initialize(serde_json::to_value(server_capabilities)?)?;
  let initialize_params = serde_json::from_value::<InitializeParams>(initialize_params)?;
  let document_changes = initialize_params
    .capabilities
    .workspace
    .and_then(|workspace| workspace.workspace_edit)
    .and_then(|workspace_edit| workspace_edit.document_changes)
    .unwrap_or_default();
  let initialization_options = initialize_params
    .initialization_options
    .ok_or_else(|| anyhow!("Missing initialization options"))?;
  let config = serde_json::from_value::<Config>(initialization_options)?;

  let templates = config
    .chat
    .messages
    .iter()
    .map(|message| Template::new(message.content.clone()).map_err(|e| e.into()))
    .collect::<Result<_>>()?;

  let mut state = State {
    document_changes,
    sender: Arc::new(connection.sender),
    client: Arc::new(reqwest::Client::new()),
    config,
    documents: Default::default(),
    tasks: Default::default(),
    templates,
  };

  for msg in &connection.receiver {
    match msg {
      Message::Request(request) => {
        if request.method == InlineCompletionRequest::METHOD {
          let (request_id, params) = request.extract::<InlineCompletionParams>(InlineCompletionRequest::METHOD)?;
          state.inline_completion_request(request_id, params)?;
        } else if request.method == ExecuteCommand::METHOD {
          let (request_id, params) = request.extract::<ExecuteCommandParams>(ExecuteCommand::METHOD)?;
          if params.command.as_str() == REWRITE_COMMAND {
            state.rewrite(request_id, params.arguments)?;
          } else {
            state.sender.send(Message::Response(LspResponse::new_err(
              request_id,
              ErrorCode::InvalidRequest as i32,
              format!("Unknown command: {}", params.command),
            )))?;
          }
        }
      }
      Message::Notification(notification) => {
        if notification.method == DidOpenTextDocument::METHOD {
          let params: DidOpenTextDocumentParams = serde_json::from_value(notification.params)?;
          state.did_open_text_document(params)?;
        } else if notification.method == DidCloseTextDocument::METHOD {
          let params: DidCloseTextDocumentParams = serde_json::from_value(notification.params)?;
          state.did_close_text_document(params);
        } else if notification.method == DidChangeTextDocument::METHOD {
          let params: DidChangeTextDocumentParams = serde_json::from_value(notification.params)?;
          state.did_change_text_document(params)?;
        } else if notification.method == Cancel::METHOD {
          let params: CancelParams = serde_json::from_value(notification.params)?;
          state.cancel(params);
        } else if notification.method == Exit::METHOD {
          return Ok(());
        }
      }
      _ => {}
    }
  }

  io_threads.join().map_err(|e| e.into())
}
