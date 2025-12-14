Configure Logfire: 

# Configure Logfire observability and instrument FastAPI latency tracing. 

# If authentication/config is missing, gracefully fall back to a no-op shim so the app still runs. 

try: 

    logfire.configure()  

    logfire.instrument_fastapi(app) 

    LOGFIRE_ENABLED = True 

except Exception as logfire_exc: 

    LOGFIRE_ENABLED = False 

    logger.warning("Logfire disabled: %s", logfire_exc) 

 

    class _LogfireNoop: 

        @staticmethod 

        def info(event_name: str, **fields: Any): 

            logger.info("logfire noop %s %s", event_name, fields) 

 

    logfire = _LogfireNoop() 

 

Functions to move the metrics into csv file: 

# Aggregate token counters per session (thread-safe) 

token_totals_lock = Lock() 

metrics_lock = Lock() 

METRICS_XLSX_PATH = Path("data") / "api_metrics.xlsx" 

METRICS_MASTER_CSV = Path("data") / "api_metrics_master.csv" 

METRICS_QUEUE_CSV = Path("data") / "api_metrics_queue.csv" 

 

# ------------------------------------------------------------------------------------ 

# Helpers 

# ------------------------------------------------------------------------------------ 

def _blank_token_totals() -> Dict[str, float]: 

    return { 

        "request_tokens": 0, 

        "response_tokens": 0, 

        "llm_prompt_tokens": 0, 

        "llm_completion_tokens": 0, 

        "llm_total_tokens": 0, 

        "total_tokens_all": 0, 

    } 

 

def estimate_tokens(text: Any) -> int: 

    """ 

    Lightweight token estimate based on whitespace splitting. 

    This avoids extra dependencies while still giving a useful relative measure. 

    """ 

    if text is None: 

        return 0 

    return len(str(text).split()) 

 

def _ensure_csv_header(path: Path, headers: List[str]) -> None: 

    if not path.exists(): 

        path.parent.mkdir(parents=True, exist_ok=True) 

        with open(path, "w", newline="", encoding="utf-8") as f: 

            csv.writer(f).writerow(headers) 

 

def _append_csv_row(path: Path, row: List[Any], headers: List[str]) -> None: 

    _ensure_csv_header(path, headers) 

    with open(path, "a", newline="", encoding="utf-8") as f: 

        csv.writer(f).writerow(row) 

 

def _read_csv_rows(path: Path) -> List[List[Any]]: 

    if not path.exists(): 

        return [] 

    try: 

        with open(path, newline="", encoding="utf-8") as f: 

            reader = list(csv.reader(f)) 

        if not reader: 

            return [] 

        header = reader[0] 

        rows = reader[1:] if header and "SessionId" in header else reader 

        return rows 

    except Exception as exc: 

        logger.warning("Failed to read CSV %s: %s", path, exc) 

        return [] 

 

def _flush_queue_to_master(headers: List[str]) -> None: 

    """ 

    Move rows from queue CSV into the master CSV when the master is writable. 

    """ 

    queued = _read_csv_rows(METRICS_QUEUE_CSV) 

    if not queued: 

        return 

    try: 

        _ensure_csv_header(METRICS_MASTER_CSV, headers) 

        with open(METRICS_MASTER_CSV, "a", newline="", encoding="utf-8") as f: 

            writer = csv.writer(f) 

            writer.writerows(queued) 

        # clear queue 

        METRICS_QUEUE_CSV.unlink(missing_ok=True) 

    except PermissionError: 

        # master locked; keep queue for later 

        return 

    except Exception as exc: 

        logger.warning("Failed to flush queue to master: %s", exc) 

 

def _sync_csv_to_xlsx(headers: List[str]) -> None: 

    """ 

    Regenerate Excel from master (and queue if master locked). 

    """ 

    # First try to flush queue into master 

    _flush_queue_to_master(headers) 

 

    # Gather rows from master 

    try: 

        master_rows = _read_csv_rows(METRICS_MASTER_CSV) 

    except Exception: 

        master_rows = [] 

 

    if not master_rows: 

        return 

 

    try: 

        METRICS_XLSX_PATH.parent.mkdir(parents=True, exist_ok=True) 

        wb = Workbook() 

        ws = wb.active 

        ws.append(headers) 

        for r in master_rows: 

            ws.append(r) 

        wb.save(METRICS_XLSX_PATH) 

        wb.close() 

    except PermissionError: 

        # Excel open/locked; try next time 

        return 

    except Exception as exc: 

        logger.warning("Failed to sync CSV to Excel: %s", exc) 

 

def _append_metrics_row(row: List[Any]): 

    """ 

    Append a single metrics row to CSV (source of truth) and best-effort sync to Excel. 

    Errors never impact API responses. 

    """ 

    headers = [ 

        "SessionId", 

        "Prompt", 

        "AIResponse", 

        "PromptTitle", 

        "TotalTokensIn", 

        "TotalTokensOut", 

        "TotalTokensAll", 

        "LatencyMs", 

        "ApiHitTime", 

    ] 

    print("headers",headers) 

 

    try: 

        _append_csv_row(METRICS_MASTER_CSV, row, headers) 

    except PermissionError: 

        # If master locked, queue it and stop. 

        try: 

            _append_csv_row(METRICS_QUEUE_CSV, row, headers) 

        except Exception as exc: 

            logger.warning("Failed to write metrics to queue: %s", exc) 

        return 

    except Exception as exc: 

        logger.warning("Failed to write metrics to master: %s", exc) 

        try: 

            _append_csv_row(METRICS_QUEUE_CSV, row, headers) 

        except Exception as exc2: 

            logger.warning("Failed to write metrics to queue after master error: %s", exc2) 

        return 

 

    # If we wrote to master, also try to flush any queued rows then update Excel. 

    _sync_csv_to_xlsx(headers) 

 

Counting the tokens and updating the metrics: 

""" 

    Main /chat endpoint wrapper with telemetry tracking and metrics persistence. 

    Calls _handle_chat_logic internally and adds token/latency metrics to response. 

    """ 

    request_start = perf_counter() 

    telemetry: Dict[str, Any] = { 

        "llm_prompt_tokens": 0, 

        "llm_completion_tokens": 0, 

        "llm_latency_ms": 0.0, 

        "thread_id": "", 

    } 

    response_payload: Dict[str, Any] = {} 

    error: Optional[Exception] = None 

 

    try: 

        set_llm_telemetry(telemetry) 

        response_payload = await _handle_chat_logic(request, telemetry) 

    except Exception as exc: 

        error = exc 

        logger.error("Error in /chat: %s\n%s", exc, traceback.format_exc()) 

        raise 

    finally: 

        set_llm_telemetry(None) 

         

        # Calculate metrics 

        total_latency_ms = round((perf_counter() - request_start) * 1000, 2) 

        response_text = "" 

        if isinstance(response_payload, dict): 

            response_text = str(response_payload.get("AIResponse", "")) 

 

        token_usage = { 

            "request_tokens": estimate_tokens(getattr(request, "message", "")), 

            "response_tokens": estimate_tokens(response_text), 

            "llm_prompt_tokens": telemetry.get("llm_prompt_tokens", 0), 

            "llm_completion_tokens": telemetry.get("llm_completion_tokens", 0), 

        } 

        token_usage["llm_total_tokens"] = token_usage["llm_prompt_tokens"] + token_usage["llm_completion_tokens"] 

        token_usage["total_tokens_all"] = ( 

            token_usage["request_tokens"] 

            + token_usage["response_tokens"] 

            + token_usage["llm_total_tokens"] 

        ) 

 

        # Update session token totals 

        token_usage_totals: Dict[str, float] = {} 

        tid = telemetry.get("thread_id") or getattr(request, "thread_id", "") 

        if tid: 

            with token_totals_lock: 

                sess = session_state.get(tid) 

                if sess is not None: 

                    totals = sess.get("token_totals") or _blank_token_totals() 

                    totals["request_tokens"] += token_usage["request_tokens"] 

                    totals["response_tokens"] += token_usage["response_tokens"] 

                    totals["llm_prompt_tokens"] += token_usage["llm_prompt_tokens"] 

                    totals["llm_completion_tokens"] += token_usage["llm_completion_tokens"] 

                    totals["llm_total_tokens"] += token_usage["llm_total_tokens"] 

                    totals["total_tokens_all"] += token_usage["total_tokens_all"] 

                    sess["token_totals"] = totals 

                    token_usage_totals = dict(totals) 

 

        # Add metrics to response 

        if isinstance(response_payload, dict): 

            response_payload["LatencyMs"] = total_latency_ms 

            response_payload["LLMLatencyMs"] = round(telemetry.get("llm_latency_ms", 0.0), 2) 

            response_payload["TokenUsage"] = token_usage 

            if token_usage_totals: 

                response_payload["TokenUsageTotals"] = token_usage_totals 

 

        # Log metrics 

        logfire.info( 

            "chat_request", 

            path=str(http_request.url.path), 

            latency_ms=total_latency_ms, 

            llm_latency_ms=round(telemetry.get("llm_latency_ms", 0.0), 2), 

            error=str(error) if error else None, 

            **token_usage, 

            **({f"total_{k}": v for k, v in token_usage_totals.items()} if token_usage_totals else {}), 

        ) 

 

        # Persist metrics to Excel (best-effort) 

        try: 

            with metrics_lock: 

                total_in = token_usage["request_tokens"] + token_usage["llm_prompt_tokens"] 

                total_out = token_usage["response_tokens"] + token_usage["llm_completion_tokens"] 

                api_hit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 

 

                # include AI response and PromptTitle in metrics row 

                ai_response = "" 

                prompt_title = "" 

                if isinstance(response_payload, dict): 

                    ai_response = str(response_payload.get("AIResponse", "")) 

                    prompt_title = str(response_payload.get("PromptTitle", "")) 

 

                # Optionally truncate long text to avoid extremely large CSV cells 

                MAX_CELL_LEN = 32700 

                if len(ai_response) > MAX_CELL_LEN: 

                    ai_response = ai_response[:MAX_CELL_LEN] 

                if len(prompt_title) > MAX_CELL_LEN: 

                    prompt_title = prompt_title[:MAX_CELL_LEN] 

 

                _append_metrics_row([ 

                    tid, 

                    getattr(request, "message", ""), 

                    ai_response, 

                    prompt_title, 

                    total_in, 

                    total_out, 

                    token_usage["total_tokens_all"], 

                    total_latency_ms, 

                    api_hit_time, 

                ]) 

        except Exception as exc: 

            logger.warning("Failed to write metrics to Excel: %s", exc) 

 

    return response_payload 

 

 

 
