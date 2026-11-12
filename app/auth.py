import secrets

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, URLSafeTimedSerializer

from app.config import settings

router = APIRouter(prefix="/auth")
COOKIE = "rc_session"
MAX_AGE = 30 * 24 * 3600


def _signer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings().session_secret)


def current_user(request: Request) -> dict | None:
    raw = request.cookies.get(COOKIE)
    if not raw:
        return None
    try:
        return _signer().loads(raw, max_age=MAX_AGE)
    except BadSignature:
        return None


def require_user(request: Request) -> dict:
    user = current_user(request)
    if not user:
        raise HTTPException(401, "sign in required")
    return user


@router.get("/login")
def login(request: Request):
    cfg = settings()
    if not cfg.github_client_id:
        raise HTTPException(503, "sign-in is not configured")
    state = secrets.token_urlsafe(16)
    url = (
        "https://github.com/login/oauth/authorize"
        f"?client_id={cfg.github_client_id}&state={state}"
        f"&redirect_uri={request.url_for('callback')}"
    )
    resp = RedirectResponse(url, status_code=302)
    resp.set_cookie("rc_state", state, max_age=600, httponly=True, samesite="lax")
    return resp


@router.get("/callback", name="callback")
async def callback(request: Request, code: str = "", state: str = ""):
    cfg = settings()
    if not code or not state or state != request.cookies.get("rc_state"):
        raise HTTPException(400, "bad oauth state")
    async with httpx.AsyncClient(timeout=15) as client:
        token = await client.post(
            "https://github.com/login/oauth/access_token",
            headers={"accept": "application/json"},
            data={
                "client_id": cfg.github_client_id,
                "client_secret": cfg.github_client_secret,
                "code": code,
                "redirect_uri": str(request.url_for("callback")),
            },
        )
        access = token.json().get("access_token")
        if not access:
            raise HTTPException(400, "github did not return a token")
        me = await client.get(
            "https://api.github.com/user",
            headers={"authorization": f"Bearer {access}", "accept": "application/json"},
        )
    gh = me.json()
    user = {"id": str(gh["id"]), "login": gh["login"], "avatar": gh.get("avatar_url", "")}
    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie(
        COOKIE,
        _signer().dumps(user),
        max_age=MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
    )
    resp.delete_cookie("rc_state")
    return resp


@router.get("/me")
def me(request: Request):
    return {"user": current_user(request), "enabled": bool(settings().github_client_id)}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE)
    return {"ok": True}
