#!/usr/bin/env python3
"""Run initial-login and forced-refresh checks in a disposable real browser."""

from __future__ import annotations

import argparse
import importlib
import json
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

APP_URL = "https://aerodocs-spike.invalid/ui/"
IDP_HOST: Final = "idp.aerodocs-spike.invalid"
MOCK_CONTROL = "http://127.0.0.1:18000"
OAUTH_CONTROL = "http://127.0.0.1:14180"


class BrowserFailure(RuntimeError):
    pass


def _mock(path: str, expected: str | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        MOCK_CONTROL + path, method="POST" if path.endswith("reset") else "GET"
    )
    if expected is not None:
        request.add_header("X-Spike-Expected-Gateway-Token", expected)
    with urllib.request.urlopen(request, timeout=3) as response:
        body = response.read(65537)
    if len(body) > 65536:
        raise BrowserFailure("control_response_too_large")
    return json.loads(body) if body else {}


def _oauth_access_token(driver: Any) -> str:
    cookie = "; ".join(
        f"{item['name']}={item['value']}" for item in driver.get_cookies()
    )
    request = urllib.request.Request(OAUTH_CONTROL + "/oauth2/auth")
    request.add_header("Cookie", cookie)
    with urllib.request.urlopen(request, timeout=5) as response:
        values = response.headers.get_all("X-Auth-Request-Access-Token") or []
    if len(values) != 1 or not values[0] or len(values[0].encode("ascii")) > 3072:
        raise BrowserFailure("invalid_auth_access_token_envelope")
    return values[0]


def _clearing_fields(driver: Any) -> tuple[int, bool, bool]:
    count = 0
    attributes_ok = True
    leaked = False
    for entry in driver.get_log("performance"):
        message = json.loads(entry["message"])["message"]
        if message["method"] != "Network.responseReceivedExtraInfo":
            continue
        text = message["params"].get("headersText", "")
        for line in text.split("\r\n"):
            if line.lower().startswith("set-cookie:"):
                count += 1
                lower = line.lower()
                attributes_ok &= "max-age=0" in lower or "expires=" in lower
            if line.lower().startswith(
                ("x-auth-request-access-token:", "x-aerodocs-gateway-access-token:")
            ):
                leaked = True
    return count, attributes_ok, leaked


def _performance(driver: Any) -> tuple[int, bool, bool, tuple[str, ...]]:
    messages = [
        json.loads(entry["message"])["message"]
        for entry in driver.get_log("performance")
    ]
    urls = {
        message["params"]["requestId"]: message["params"]["response"]["url"]
        for message in messages
        if message["method"] == "Network.responseReceived"
    }
    set_cookie_fields = 0
    access_header_seen = False
    cacheable_auth_response = False
    ordered_parts: list[str] = []
    for message in messages:
        if message["method"] != "Network.responseReceivedExtraInfo":
            continue
        headers = message["params"].get("headers", {})
        lowered = {str(name).lower(): str(value) for name, value in headers.items()}
        if (
            "x-auth-request-access-token" in lowered
            or "x-aerodocs-gateway-access-token" in lowered
        ):
            access_header_seen = True
        header_text = message["params"].get("headersText", "")
        set_cookie_fields += sum(
            line.lower().startswith("set-cookie:") for line in header_text.split("\r\n")
        )
        ordered_parts.extend(
            line.split(":", 1)[1].strip().split("=", 1)[0]
            for line in header_text.split("\r\n")
            if line.lower().startswith("set-cookie:")
        )
        cache_control = lowered.get("cache-control", "").lower()
        url = urls.get(message["params"]["requestId"], "")
        auth_derived = (
            url.startswith("https://idp.aerodocs-spike.invalid/") or "/oauth2/" in url
        )
        if auth_derived and "no-store" not in cache_control:
            cacheable_auth_response = True
    return (
        set_cookie_fields,
        access_header_seen,
        cacheable_auth_response,
        tuple(ordered_parts),
    )


def _login_return_target_seen(driver: Any) -> bool:
    for entry in driver.get_log("performance"):
        message = json.loads(entry["message"])["message"]
        if message["method"] != "Network.requestWillBeSent":
            continue
        url = str(message["params"]["request"].get("url", ""))
        if url in {
            "https://aerodocs-spike.invalid/oauth2/start?rd=/ui/",
            "https://aerodocs-spike.invalid/oauth2/start?rd=%2Fui%2F",
        }:
            return True
    return False


def _cookie_metrics(driver: Any) -> tuple[int, int, bool]:
    cookies = [
        cookie for cookie in driver.get_cookies() if "aerodocs_spike" in cookie["name"]
    ]
    total = sum(len(cookie["value"].encode("utf-8")) for cookie in cookies)
    secure = all(
        cookie.get("secure") is True
        and cookie.get("httpOnly") is True
        and str(cookie.get("sameSite", "")).lower() == "lax"
        and cookie.get("path") == "/"
        and isinstance(cookie.get("expiry"), (int, float))
        for cookie in cookies
    )
    return len(cookies), total, secure


def run(
    username_file: Path, password_file: Path, profile_dir: Path
) -> dict[str, int | bool]:
    try:
        webdriver = importlib.import_module("selenium.webdriver")
        by_module = importlib.import_module("selenium.webdriver.common.by")
        wait_module = importlib.import_module("selenium.webdriver.support.ui")
    except ImportError as exc:
        raise BrowserFailure("selenium_unavailable") from exc
    by = by_module.By
    web_driver_wait = wait_module.WebDriverWait

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-first-run")
    options.add_argument("--disable-background-networking")
    options.add_argument("--disable-sync")
    options.add_argument(f"--user-data-dir={profile_dir}")
    options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(APP_URL)
        web_driver_wait(driver, 10).until(
            lambda current: IDP_HOST in current.current_url
        )
        if not _login_return_target_seen(driver):
            raise BrowserFailure("login_return_target")
        driver.find_element(by.ID, "username").send_keys(
            username_file.read_text(encoding="utf-8").strip()
        )
        driver.find_element(by.ID, "password").send_keys(
            password_file.read_text(encoding="utf-8").strip()
        )
        driver.find_element(by.ID, "kc-login").click()
        web_driver_wait(driver, 15).until(
            lambda current: current.current_url == APP_URL
        )
        if (
            driver.find_element(by.TAG_NAME, "body").text.strip()
            != '{"mock":"AERODOCS_RESPONSE_SENTINEL"}'
        ):
            raise BrowserFailure("upstream_response_changed")
        expected_token = _oauth_access_token(driver)
        _mock("/__control/reset", expected_token)
        driver.get(APP_URL)
        snapshot = _mock("/__control/snapshot")
        last = snapshot.get("last", {})
        if (
            not isinstance(last, dict)
            or last.get("gateway_header_count") != 1
            or last.get("gateway_token_bytes", 0) == 0
            or not last.get("gateway_token_matches_expected")
            or last.get("browser_cookie_field_count") != 0
        ):
            raise BrowserFailure("browser_data_plane")
        before = {cookie["name"]: cookie["value"] for cookie in driver.get_cookies()}
        parts, values, secure = _cookie_metrics(driver)
        if parts < 1 or parts > 3 or values > 10240 or not secure:
            raise BrowserFailure("session_cookie_limits")
        initial_fields, initial_leak, initial_cacheable, initial_order = _performance(
            driver
        )
        if initial_fields < 1 or len(initial_order) != initial_fields:
            raise BrowserFailure("initial_cookie_fields_missing_or_collapsed")
        time.sleep(65)
        driver.get(APP_URL)
        web_driver_wait(driver, 15).until(
            lambda current: current.current_url == APP_URL
        )
        after = {cookie["name"]: cookie["value"] for cookie in driver.get_cookies()}
        refresh_fields, refresh_leak, refresh_cacheable, refresh_order = _performance(
            driver
        )
        if before == after:
            raise BrowserFailure("refresh_not_observed")
        if refresh_fields < 1 or len(refresh_order) != refresh_fields:
            raise BrowserFailure("refresh_cookie_fields_missing_or_collapsed")
        if initial_leak or refresh_leak or initial_cacheable or refresh_cacheable:
            raise BrowserFailure("browser_header_leak")
        driver.execute_cdp_cmd(
            "Network.setExtraHTTPHeaders", {"headers": {"X-Spike-Long-Response": "45"}}
        )
        started = time.monotonic()
        driver.get(APP_URL)
        long_seconds = time.monotonic() - started
        driver.execute_cdp_cmd("Network.setExtraHTTPHeaders", {"headers": {}})
        if (
            not 45 <= long_seconds <= 55
            or driver.find_element(by.TAG_NAME, "body").text.strip()
            != '{"mock":"AERODOCS_RESPONSE_SENTINEL"}'
        ):
            raise BrowserFailure("long_response_failed")
        _mock("/__control/reset")
        driver.get("https://aerodocs-spike.invalid/oauth2/sign_out")
        clearing_fields, clearing_attributes, clearing_leak = _clearing_fields(driver)
        stale = any(
            "aerodocs_spike" in cookie["name"] for cookie in driver.get_cookies()
        )
        after_clear = _mock("/__control/snapshot")
        if (
            clearing_fields < 1
            or not clearing_attributes
            or clearing_leak
            or stale
            or after_clear.get("invocations") != 0
        ):
            raise BrowserFailure("clearing_refresh_failed")
        return {
            "initial_set_cookie_fields": initial_fields,
            "refresh_set_cookie_fields": refresh_fields,
            "session_cookie_parts": parts,
            "session_cookie_value_bytes": values,
            "security_attributes_preserved": secure,
            "access_token_header_absent": True,
            "refresh_cookie_changed": True,
            "initial_cookie_order_preserved": len(initial_order) == initial_fields,
            "refresh_cookie_order_preserved": len(refresh_order) == refresh_fields,
            "clearing_set_cookie_fields": clearing_fields,
            "clearing_attributes_preserved": clearing_attributes,
            "stale_cookie_parts_absent": not stale,
            "mock_not_invoked_after_clear": True,
            "long_response_seconds": int(long_seconds),
        }
    finally:
        driver.quit()


def main() -> None:
    global APP_URL, MOCK_CONTROL, OAUTH_CONTROL
    parser = argparse.ArgumentParser()
    parser.add_argument("--username-file", type=Path, required=True)
    parser.add_argument("--password-file", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--endpoints-file", type=Path, required=True)
    args = parser.parse_args()
    endpoints = json.loads(args.endpoints_file.read_text(encoding="utf-8"))
    APP_URL = str(endpoints["caddy_url"]).rstrip("/") + "/ui/"
    MOCK_CONTROL = str(endpoints["mock_control_url"]).rstrip("/")
    OAUTH_CONTROL = str(endpoints["oauth_control_url"]).rstrip("/")
    if (
        APP_URL != "https://aerodocs-spike.invalid/ui/"
        or not MOCK_CONTROL.startswith("http://127.0.0.1:")
        or not OAUTH_CONTROL.startswith("http://127.0.0.1:")
    ):
        raise SystemExit(2)
    try:
        measurements = run(args.username_file, args.password_file, args.profile_dir)
    except (BrowserFailure, OSError, ValueError, json.JSONDecodeError):
        print(
            json.dumps(
                {
                    "schema_version": 2,
                    "test_id": "browser_flow",
                    "observed_at": datetime.now(UTC)
                    .replace(microsecond=0)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "result": "FAIL",
                    "measurements": {"checked": True},
                    "failure_code": "BROWSER",
                    "gate_results": {
                        "initial_login": {
                            "result": "FAIL",
                            "measurements": {"checked": True},
                        }
                    },
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from None
    gates = (
        "initial_login",
        "forced_refresh",
        "clearing_refresh",
        "long_running_response",
        "data_plane_separation",
    )
    print(
        json.dumps(
            {
                "schema_version": 2,
                "test_id": "browser_flow",
                "observed_at": datetime.now(UTC)
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z"),
                "result": "PASS",
                "measurements": measurements,
                "failure_code": "NONE",
                "gate_results": {
                    gate: {"result": "PASS", "measurements": {"checked": True}}
                    for gate in gates
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
