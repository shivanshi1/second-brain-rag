from pathlib import Path

import httpx
import typer

from secondbrain.ingestion.loaders import EXTENSION_TO_TYPE

app = typer.Typer(help="Second Brain RAG command line.")
API = typer.Option("http://localhost:8000", envvar="SECONDBRAIN_API", help="API base URL")
KEY = typer.Option("", envvar="SECONDBRAIN_API_KEY", help="API key, if enabled")


@app.command()
def ingest(
    paths: list[Path] = typer.Argument(..., exists=True, help="Files or folders"),
    tags: str = typer.Option("", help="Comma-separated tags for every file"),
    api: str = API,
    key: str = KEY,
) -> None:
    """Upload files (folders are walked recursively)."""
    files = [
        f
        for p in paths
        for f in ([p] if p.is_file() else sorted(p.rglob("*")))
        if f.is_file() and f.suffix.lower() in EXTENSION_TO_TYPE
    ]
    headers = {"X-API-Key": key} if key else {}
    for f in files:
        with f.open("rb") as fh:
            r = httpx.post(
                f"{api}/documents",
                files={"file": (f.name, fh)},
                data={"tags": tags},
                headers=headers,
                timeout=60,
            )
        r.raise_for_status()
        typer.echo(f"{r.json()['status']:<10} {f}")
    typer.echo(f"{len(files)} file(s) submitted.")


@app.command()
def search(query: str, top_k: int = 5, api: str = API, key: str = KEY) -> None:
    """Hybrid search over your documents."""
    headers = {"X-API-Key": key} if key else {}
    r = httpx.post(
        f"{api}/search", json={"query": query, "top_k": top_k}, headers=headers, timeout=60
    )
    r.raise_for_status()
    for hit in r.json():
        where = hit["heading_path"] or (f"p.{hit['page_number']}" if hit["page_number"] else "")
        typer.echo(f"\n[{hit['score']:.3f}] {hit['document_title']} {where}\n{hit['text'][:300]}")


@app.command()
def ask(
    question: str,
    top_k: int = typer.Option(3, min=1, max=20),
    strict: bool = typer.Option(True, "--strict/--open", help="Answer only from retrieved notes."),
    api: str = API,
    key: str = KEY,
) -> None:
    """Ask a question across your notes and print the answer with source citations."""
    headers = {"X-API-Key": key} if key else {}
    r = httpx.post(
        f"{api}/ask",
        json={"question": question, "top_k": top_k, "strict": strict},
        headers=headers,
        timeout=180,
    )
    r.raise_for_status()
    result = r.json()
    typer.echo(result["answer"])
    typer.echo(f"\nCitations valid: {result['citations_valid']}")
    typer.echo(f"Sentence support: {result['supported_ratio']:.0%}")
    for citation in result["citations"]:
        where = f" > {citation['heading']}" if citation["heading"] else ""
        typer.echo(f"\n[{citation['n']}] {citation['document']}{where}\n{citation['snippet']}")


if __name__ == "__main__":
    app()
