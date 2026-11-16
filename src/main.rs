use axum::{
    Json, Router,
    extract::Path,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "crab=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health))
        .route("/hello/{name}", get(hello))
        .route("/echo", post(echo))
        .layer(TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .expect("failed to bind 127.0.0.1:3000");

    tracing::info!("listening on http://{}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.expect("server error");
}

async fn root() -> &'static str {
    "crab is running"
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(Health { status: "ok" }))
}

async fn hello(Path(name): Path<String>) -> impl IntoResponse {
    Json(HelloResponse {
        message: format!("hello, {name}"),
    })
}

async fn echo(Json(payload): Json<EchoRequest>) -> impl IntoResponse {
    Json(payload)
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
}

#[derive(Serialize)]
struct HelloResponse {
    message: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct EchoRequest {
    message: String,
}
