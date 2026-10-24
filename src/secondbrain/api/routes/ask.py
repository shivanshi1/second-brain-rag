from fastapi import APIRouter, HTTPException

from secondbrain.schemas import AskRequest, AskResponse

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """TODO (Phase 2/3). Suggested steps:

    1. hits = hybrid_search(db, req.question, req.filters, req.top_k)
    2. (Phase 3) rerank hits; if strict and best score < threshold -> refuse
    3. messages = build_messages(req.question, hits, req.strict)
    4. answer = get_llm().generate(messages)
    5. validate citation markers map to real blocks -> set grounded flag
    6. write a QueryLog row; return AskResponse with citations and latency
    """
    raise HTTPException(501, "Not implemented yet: see the docstring for the plan.")
