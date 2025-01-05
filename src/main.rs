mod fim;
mod mistral;

use std::{collections::HashMap, env, fs::File, io::BufReader, sync::Arc};

use anyhow::Result;
use fim::Fim;
use lsp_server::{Connection, Message, RequestId, Response};
use lsp_types::{
  notification::{Cancel, DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Notification},
  request::{InlineCompletionRequest, Request, Shutdown},
  CancelParams, DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
  InlineCompletionItem, InlineCompletionParams, InlineCompletionResponse, NumberOrString, OneOf, Range,
  ServerCapabilities, TextDocumentSyncKind, Uri,
};
use mistral::fim::MistralFim;
use ropey::Rope;
use tokio::task::JoinHandle;

#[derive(Debug, Default)]
struct State {
  documents: HashMap<Uri, Rope>,
  tasks: Arc<papaya::HashMap<RequestId, JoinHandle<()>>>,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
  let env = env_logger::Env::default().filter_or("RUST_LOG", "info");
  env_logger::Builder::from_env(env).init();

  let mistral_api_key = env::var("MISTRAL_API_KEY").unwrap();

  let mistral_fim = Arc::new(MistralFim {
    url: "https://api.mistral.ai/v1/fim/completions".to_string(),
    api_key: mistral_api_key,
    model: "codestral-latest".to_string(),
    temperature: Some(0.0),
    top_p: None,
    max_tokens: Some(256),
    min_tokens: None,
    stop: Some("\n\n".to_string()),
    random_seed: None,
  });

  let (connection, io_threads) = Connection::stdio();
  let server_capabilities = ServerCapabilities {
    inline_completion_provider: Some(OneOf::Left(true)),
    text_document_sync: Some(lsp_types::TextDocumentSyncCapability::Kind(
      TextDocumentSyncKind::INCREMENTAL,
    )),
    ..Default::default()
  };
  let _initialization_args = connection.initialize(serde_json::to_value(server_capabilities)?)?;

  let sender = Arc::new(connection.sender);
  let client = Arc::new(reqwest::Client::new());

  let mut state = State::default();

  for msg in &connection.receiver {
    match msg {
      Message::Request(request) => {
        if request.method == Shutdown::METHOD {
          return Ok(());
        } else if request.method == InlineCompletionRequest::METHOD {
          let (request_id, params) = request.extract::<InlineCompletionParams>(InlineCompletionRequest::METHOD)?;
          let rope = state
            .documents
            .get(&params.text_document_position.text_document.uri)
            .expect("missing document");

          let index = rope.line_to_char(params.text_document_position.position.line as usize)
            + params.text_document_position.position.character as usize;
          let prefix = rope.slice(0..index).to_string();
          let suffix = rope.slice(index..rope.len_chars()).to_string();

          let mistral_fim = mistral_fim.clone();
          let client = client.clone();
          let sender = sender.clone();
          let tasks = state.tasks.clone();
          let request_id_c = request_id.clone();
          let future = async move {
            let completions = mistral_fim.fim(client, prefix, suffix).await;
            match completions {
              Result::Ok(completions) => {
                let range = Range::new(
                  params.text_document_position.position,
                  params.text_document_position.position,
                );
                let completion_items = completions
                  .map(|completion| InlineCompletionItem {
                    insert_text: completion,
                    filter_text: None,
                    range: Some(range),
                    command: None,
                    insert_text_format: None,
                  })
                  .collect();
                tasks.pin().remove(&request_id_c);
                if let Result::Err(error) = sender.send(Message::Response(Response::new_ok(
                  request_id_c,
                  InlineCompletionResponse::Array(completion_items),
                ))) {
                  log::warn!("Failed to send server response: {}", error);
                }
              }
              Result::Err(error) => {
                tasks.pin().remove(&request_id_c);
                log::warn!("Failed to get mistral response: {}", error);
              }
            }
          };
          let handle = tokio::task::spawn(future);
          state.tasks.pin().insert(request_id, handle);
        }
      }
      Message::Notification(notification) => {
        if notification.method == DidOpenTextDocument::METHOD {
          let params: DidOpenTextDocumentParams = serde_json::from_value(notification.params)?;
          let file = File::open(params.text_document.uri.path().as_str())?;
          let reader = BufReader::new(file);
          let rope = Rope::from_reader(reader)?;
          state.documents.insert(params.text_document.uri, rope);
        } else if notification.method == DidCloseTextDocument::METHOD {
          let params: DidCloseTextDocumentParams = serde_json::from_value(notification.params)?;
          state.documents.remove(&params.text_document.uri);
        } else if notification.method == DidChangeTextDocument::METHOD {
          let params: DidChangeTextDocumentParams = serde_json::from_value(notification.params)?;
          for change in params.content_changes {
            if let Some(range) = change.range {
              let rope = state
                .documents
                .get_mut(&params.text_document.uri)
                .expect("missing document");
              let start_index = rope.line_to_char(range.start.line as usize) + range.start.character as usize;
              let end_index = rope.line_to_char(range.end.line as usize) + range.end.character as usize;
              rope.remove(start_index..end_index);
              rope.insert(start_index, &change.text);
            } else {
              let rope = Rope::from_str(&change.text);
              state.documents.insert(params.text_document.uri.clone(), rope);
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
        }
      }
      _ => {}
    }
  }

  io_threads.join().map_err(|e| e.into())
}
