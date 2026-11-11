mod config;
mod copilot;
mod infill;
mod llama_cpp;
mod mistral;
mod ollama;
mod openai;

use std::sync::Arc;

use anyhow::{Result, anyhow};
use clap::Command;
use config::Config;
use crossbeam_channel::Sender;
use dashmap::DashMap;
use infill::Infill;
use lsp_server::{Connection, ErrorCode, Message, RequestId, Response as LspResponse};
use lsp_types::{
  CancelParams, DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams, InitializeParams,
  InlineCompletionItem, InlineCompletionParams, InlineCompletionResponse, NumberOrString, OneOf, Position, Range,
  ServerCapabilities, TextDocumentSyncKind, Uri,
  notification::{Cancel, DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, Exit, Notification},
  request::{InlineCompletionRequest, Request, Shutdown},
};

use reqwest::Client;
use ropey::Rope;
use tokio::task::JoinHandle;

fn position_to_char_index(rope: &Rope, line: u32, character: u32) -> usize {
  let line_char_start = rope.line_to_char(line as usize);
  let target_utf16 = rope.char_to_utf16_cu(line_char_start) + character as usize;
  if target_utf16 >= rope.len_utf16_cu() {
    rope.len_chars()
  } else {
    let char_idx = rope.utf16_cu_to_char(target_utf16);
    if rope.char_to_utf16_cu(char_idx) < target_utf16 {
      char_idx + 1
    } else {
      char_idx
    }
  }
}

#[derive(Debug)]
struct Document {
  rope: Rope,
  version: i32,
  language_id: String,
}

#[derive(Debug)]
struct State {
  sender: Arc<Sender<Message>>,
  client: Arc<Client>,
  config: Config,
  documents: Arc<DashMap<Uri, Document>>,
  tasks: Arc<DashMap<RequestId, JoinHandle<Result<()>>>>,
  pending_requests: Arc<DashMap<Uri, RequestId>>,
}

impl State {
  #[allow(clippy::collapsible_if)]
  fn inline_completion_request(&self, request_id: RequestId, params: InlineCompletionParams) -> Result<()> {
    let document_uri = params.text_document_position.text_document.uri.clone();

    // Cancel any in-flight completion request for the same document
    if let Some(pending_id) = self.pending_requests.get(&document_uri).map(|r| r.clone()) {
      if let Some((_, handle)) = self.tasks.remove(&pending_id) {
        handle.abort();
      }
    }
    self.pending_requests.insert(document_uri.clone(), request_id.clone());

    let pending_requests = self.pending_requests.clone();
    let document_uri = document_uri.clone();

    let document = self
      .documents
      .get(&document_uri)
      .ok_or_else(|| anyhow!("Missing document: {}", document_uri.as_str()))?;

    let line = params.text_document_position.position.line;
    let character = params.text_document_position.position.character;
    let index = position_to_char_index(&document.rope, line, character);
    let prefix = document.rope.slice(..index).to_string();
    let suffix = document.rope.slice(index..).to_string();
    let language_id = document.language_id.clone();

    let infill = self.config.get_infill();
    let client = self.client.clone();
    let sender = self.sender.clone();
    let tasks = self.tasks.clone();
    let request_id_c = request_id.clone();
    let future = async move {
      let completions = infill.infill(client, prefix, suffix, language_id).await;
      match completions {
        Ok(completions) => {
          tasks.remove(&request_id_c);
          pending_requests.remove_if(&document_uri, |_, v| *v == request_id_c);
          let range = Range::new(Position { line, character }, Position { line, character });
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
        Err(error) => {
          tasks.remove(&request_id_c);
          pending_requests.remove_if(&document_uri, |_, v| *v == request_id_c);
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

  fn did_open_text_document(&mut self, params: DidOpenTextDocumentParams) -> Result<()> {
    let rope = Rope::from_str(&params.text_document.text);
    self.documents.insert(
      params.text_document.uri,
      Document {
        rope,
        version: params.text_document.version,
        language_id: params.text_document.language_id,
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
        let start_index = position_to_char_index(&document.rope, range.start.line, range.start.character);
        let end_index = position_to_char_index(&document.rope, range.end.line, range.end.character);
        document.rope.remove(start_index..end_index);
        document.rope.insert(start_index, &change.text);
        document.version = params.text_document.version;
      } else {
        let rope = Rope::from_str(&change.text);
        let language_id = self
          .documents
          .get(&params.text_document.uri)
          .map(|d| d.language_id.clone())
          .unwrap_or_default();
        self.documents.insert(
          params.text_document.uri.clone(),
          Document {
            rope,
            version: params.text_document.version,
            language_id,
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
      let _ = self.sender.send(Message::Response(LspResponse::new_err(
        id,
        ErrorCode::RequestCanceled as i32,
        "Request cancelled".to_string(),
      )));
    }
  }
}

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
    execute_command_provider: None,
    inline_completion_provider: Some(OneOf::Left(true)),
    text_document_sync: Some(lsp_types::TextDocumentSyncCapability::Kind(
      TextDocumentSyncKind::INCREMENTAL,
    )),
    ..Default::default()
  };
  let initialize_params = connection.initialize(serde_json::to_value(server_capabilities)?)?;
  let initialize_params = serde_json::from_value::<InitializeParams>(initialize_params)?;

  let initialization_options = initialize_params
    .initialization_options
    .ok_or_else(|| anyhow!("Missing initialization options"))?;
  let config = serde_json::from_value::<Config>(initialization_options)?;

  let mut state = State {
    sender: Arc::new(connection.sender),
    client: Arc::new(
      reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?,
    ),
    config,
    documents: Default::default(),
    tasks: Default::default(),
    pending_requests: Default::default(),
  };

  for msg in &connection.receiver {
    match msg {
      Message::Request(request) => {
        if request.method == InlineCompletionRequest::METHOD {
          let request_id = request.id.clone();
          let (req_id, params) = match request.extract::<InlineCompletionParams>(InlineCompletionRequest::METHOD) {
            Ok(result) => result,
            Err(e) => {
              let _ = state.sender.send(Message::Response(LspResponse::new_err(
                request_id,
                ErrorCode::InternalError as i32,
                e.to_string(),
              )));
              continue;
            }
          };
          if let Err(e) = state.inline_completion_request(req_id, params) {
            let _ = state.sender.send(Message::Response(LspResponse::new_err(
              request_id,
              ErrorCode::InternalError as i32,
              e.to_string(),
            )));
          }
        } else if request.method == Shutdown::METHOD {
          let _ = state.sender.send(Message::Response(LspResponse::new_ok(
            request.id,
            serde_json::Value::Null,
          )));
          return Ok(());
        }
      }
      Message::Notification(notification) => {
        let result = match notification.method.as_str() {
          DidOpenTextDocument::METHOD => {
            let params: DidOpenTextDocumentParams = serde_json::from_value(notification.params)?;
            state.did_open_text_document(params)
          }
          DidCloseTextDocument::METHOD => {
            let params: DidCloseTextDocumentParams = serde_json::from_value(notification.params)?;
            state.did_close_text_document(params);
            Ok(())
          }
          DidChangeTextDocument::METHOD => {
            let params: DidChangeTextDocumentParams = serde_json::from_value(notification.params)?;
            state.did_change_text_document(params)
          }
          Cancel::METHOD => {
            let params: CancelParams = serde_json::from_value(notification.params)?;
            state.cancel(params);
            Ok(())
          }
          Exit::METHOD => {
            return Ok(());
          }
          _ => Ok(()),
        };
        if let Err(e) = result {
          log::warn!("Notification handler error: {}", e);
        }
      }
      _ => {}
    }
  }

  io_threads.join().map_err(|e| e.into())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::config::CompletionConfig;
  use lsp_types::Uri;
  use lsp_types::{TextDocumentContentChangeEvent, TextDocumentItem, VersionedTextDocumentIdentifier};

  #[test]
  fn test_position_to_char_index_ascii() {
    let rope = Rope::from_str("hello\nworld");
    assert_eq!(position_to_char_index(&rope, 0, 5), 5);
    assert_eq!(position_to_char_index(&rope, 1, 3), 9);
  }

  #[test]
  fn test_position_to_char_index_multibyte() {
    let rope = Rope::from_str("héllo\nworld");
    assert_eq!(position_to_char_index(&rope, 0, 5), 5);
  }

  #[test]
  fn test_position_to_char_index_emoji() {
    let rope = Rope::from_str("foo🔥bar");
    assert_eq!(position_to_char_index(&rope, 0, 4), 4);
  }

  #[test]
  fn test_position_to_char_index_beyond_end() {
    let rope = Rope::from_str("hello");
    assert_eq!(position_to_char_index(&rope, 0, 100), 5);
  }

  #[test]
  fn test_position_to_char_index_out_of_bounds_line() {
    let rope = Rope::from_str("hello\nworld");
    assert_eq!(position_to_char_index(&rope, 0, 0), 0);
  }

  #[test]
  fn test_did_change_text_document() {
    let mut state = State {
      sender: Arc::new(crossbeam_channel::unbounded().0),
      client: Arc::new(reqwest::Client::builder().build().unwrap()),
      config: Config {
        infill: CompletionConfig::default(),
      },
      documents: Arc::new(DashMap::new()),
      tasks: Arc::new(DashMap::new()),
      pending_requests: Arc::new(DashMap::new()),
    };
    let uri = "file:///test.txt".parse::<Uri>().unwrap();
    state
      .did_open_text_document(DidOpenTextDocumentParams {
        text_document: TextDocumentItem {
          uri: uri.clone(),
          language_id: "text".to_string(),
          version: 1,
          text: "hello world".to_string(),
        },
      })
      .unwrap();
    assert!(state.documents.contains_key(&uri));

    // Replace "world" (range 6..11) with "rust"
    state
      .did_change_text_document(DidChangeTextDocumentParams {
        text_document: VersionedTextDocumentIdentifier {
          uri: uri.clone(),
          version: 2,
        },
        content_changes: vec![TextDocumentContentChangeEvent {
          range: Some(Range::new(
            Position { line: 0, character: 6 },
            Position { line: 0, character: 11 },
          )),
          range_length: None,
          text: "rust".to_string(),
        }],
      })
      .unwrap();

    let doc = state.documents.get(&uri).unwrap();
    assert_eq!(doc.rope.to_string(), "hello rust");
    assert_eq!(doc.version, 2);
  }
}
