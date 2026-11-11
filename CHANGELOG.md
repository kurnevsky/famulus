# Changelog

## 0.0.6

* Fix UTF-16 position handling for inline completions
* Use text content from `didOpen` params instead of reading from disk
* Send error response when requests are cancelled
* Add per-document in-flight request suppression
* Add HTTP request timeouts
* Add notification handler error handling
* Add tests for position conversion and document sync

## 0.0.5

* Add generic OpenAI-compatible completions API
* Remove `rewrite` command

## 0.0.4

* Add `rewrite` command
* Unify generation config across all providers
* Fix typo in the config (privider -> provider)

## 0.0.3

* Make `stop` config parameter optional
* Support for Ollama generate API

## 0.0.2

* Complete support for llama-cpp and Mistral APIs

## 0.0.1

* Initial release
