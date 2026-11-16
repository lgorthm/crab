//! Crab library crate.
//!
//! The binary only wires the HTTP server; the DeepSeek Responses client lives
//! here so it can be tested and reused without starting the server.

pub mod llm;
