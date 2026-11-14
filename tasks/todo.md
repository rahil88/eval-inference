# eval_inference: FP16 vs 4-bit, accuracy and speed

**Goal:** Run Qwen2.5-1.5B-Instruct at FP16 and at Q4_K_M with llama.cpp. Score both on the same 50 GSM8K questions and compare accuracy, tokens/sec, file size and memory.

**Rules for a fair comparison:** use the same prompt, `temperature 0`, the same `max_tokens`, the same context size, all layers on GPU, one request at a time, a discarded warm-up request, the laptop plugged in and other heavy apps closed.

**Machine:** M2 (4P+4E cores), 16 GB RAM, about 100 GB/s memory bandwidth, about 16 GB free disk.

Each step has a **Do** list, a **Check** (what you should see) and a **Learn** item (a question to answer in your notes). Commit after each step.

---

## Step 0: Repo setup
- [ ] **Do:** `cd ~/Documents/Projects/eval_inference && git init`. Your home folder `~` is itself a git repo, so without this your commits would land in the wrong place.
- [ ] **Do:** create `.gitignore` with `models/`, `.venv/`, `__pycache__/` and `.DS_Store`.
- [ ] **Do:** `uv init --python 3.12`, then `uv add httpx` and `uv add --dev pytest`.
- [ ] **Do (optional):** `gh repo create eval-inference --private --source . --push`, or use `--public`. Your choice.
- [ ] **Check:** `git rev-parse --show-toplevel` prints the eval_inference path, not `/Users/craddy-san`.

## Step 1: Install llama.cpp
- [ ] **Do:** `brew install llama.cpp`
- [ ] **Check:** `llama-cli --version`, `llama-server --help`, `llama-bench --help` and `llama-quantize --help` all run.
- [ ] **Learn:** what is GGUF? Why does llama.cpp want a single file holding the weights, the tokenizer and the chat template?

## Step 2: Get the models (download FP16, quantize it yourself)
- [ ] **Do:** `hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-fp16.gguf --local-dir models`
- [ ] **Predict before running:** 1.5B parameters at 16 bits each is X GB, and at about 4.5 bits each it's Y GB. Write both numbers down.
- [ ] **Do:** `llama-quantize models/qwen2.5-1.5b-instruct-fp16.gguf models/qwen2.5-1.5b-instruct-q4_k_m.gguf Q4_K_M`
- [ ] **Check:** `ls -lh models/` shows about 3.6 GB and about 1.1 GB. Compare these with your predictions.
- [ ] **Learn:** Qwen released the weights in BF16. Why is FP16 "full precision" here, and why would FP32 gain nothing? What do "K" and "M" mean in Q4_K_M? (Look up k-quants and block-wise quantization with per-block scales.)

## Step 3: Smoke test
- [ ] **Do:** `llama-cli -m models/<file>.gguf -ngl 99 -p "What is 17*23?"` for each model.
- [ ] **Check:** both answer sensibly. Find the speed numbers llama.cpp prints at the end.
- [ ] **Learn:** what does `-ngl 99` do? What changes if you set it to 0, which keeps everything on the CPU?

## Step 4: Synthetic speed benchmark with llama-bench
- [ ] **Predict first:** generating each token means reading every weight once, so the speed ceiling is bandwidth ÷ model size. Work out the ceiling for both files at 100 GB/s.
- [ ] **Do:** `llama-bench -m models/qwen2.5-1.5b-instruct-fp16.gguf -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf -p 512 -n 128`
- [ ] **Check:** you get a table with `pp512` (prompt processing) and `tg128` (token generation) in tokens/sec. Save it to `results/llama-bench.md`.
- [ ] **Learn:** which of pp and tg sped up more from 4-bit, and why? (The terms to look up are compute-bound and memory-bandwidth-bound.) How close did tg get to your ceiling?

## Step 5: Build the 50-question set
- [ ] **Do:** download `https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl`, which has 1319 rows.
- [ ] **Do:** write `scripts/make_dataset.py`. It picks 50 rows with a fixed seed (`random.Random(42)`), takes the gold number from after `####` and strips commas, then writes `data/questions.jsonl` with `{id, question, gold}`.
- [ ] **Check:** 50 lines, every `gold` parses as a number, and running the script again gives the same file. Commit this file.
- [ ] **Learn:** why use a fixed seed, and why commit the dataset instead of making a new sample each run?

## Step 6: Serve a model and inspect the API by hand
- [ ] **Do:** `llama-server -m models/<file>.gguf -ngl 99 -c 4096 -np 1 --port 8090`. Port 8080 is already taken on this Mac. `-np 1` stops the context being split across parallel slots.
- [ ] **Do:** `curl` one question to `http://localhost:8090/v1/chat/completions` with `temperature: 0`, then pretty-print the JSON.
- [ ] **Check:** find where the answer text is and where the token counts and speeds are. Look for a `timings` object. Also call `GET /v1/models` and see which model it reports.
- [ ] **Learn:** this endpoint applies the chat template from the GGUF for you. What would break if you sent raw text to `/completion` instead?

## Step 7: Answer extractor (write the tests first)
- [ ] **Do:** design the prompt, for example "Solve step by step. End with a final line: `Answer: <number>`."
- [ ] **Do:** write `extract_answer(text) -> str | None`. It should take the `Answer:` line first and fall back to the last number in the text. It should also handle `$`, commas, a trailing `.`, `18.0` meaning `18`, and negative numbers.
- [ ] **Do:** write `tests/test_extract.py` with at least 8 cases before the function exists, and run `uv run pytest`.
- [ ] **Check:** all tests pass.
- [ ] **Learn:** more evals break here than anywhere else. If accuracy comes out suspiciously low, check the parser before blaming the model.

## Step 8: The harness
- [ ] **Do:** write `eval.py --label fp16 --out results/fp16.jsonl`, and make it run any model the same way. For each question it should:
  - send one request with `temperature 0` and `max_tokens 512`, through an `httpx.Client` with a long timeout;
  - write one JSONL line per question with `id, gold, pred, correct, output, prompt_tokens, gen_tokens, prompt_tps, gen_tps, wall_s`;
  - append and flush after every question, so a crash at question 37 keeps the first 36;
  - check `/v1/models` at startup and store the model name, so results can't be mislabeled;
  - send one warm-up request first and not record it.
- [ ] **Check:** run it with `--limit 3` first and read the 3 outputs by eye.

## Step 9: Full runs
- [ ] **Do:** start the server with FP16 and run all 50. Stop it, start it with Q4_K_M using the exact same flags, and run all 50 again.
- [ ] **Do:** while each one runs, note the server's memory use from Activity Monitor (or `ps -o rss`).
- [ ] **Check:** both files have 50 lines. Expect each run to take a few minutes.

## Step 10: Analyze
- [ ] **Do:** write `summarize.py`. Per model, it reports accuracy, gen tok/s (mean and median), prompt tok/s, mean output length, file size and RAM. Write the result to `results/summary.md`.
- [ ] **Do:** make a paired table of both right, FP16 only, Q4 only and both wrong. Read every question the two models disagree on.
- [ ] **Learn 1:** the average of per-request tok/s and (total tokens ÷ total time) give different numbers. Which is right for your question, and why?
- [ ] **Learn 2:** with n=50, the standard error is about √(p(1−p)/50), roughly ±7 points. Is the accuracy gap you found real or noise? (Look up McNemar's test for paired results.)
- [ ] **Learn 3:** do the Q4 answers get longer or shorter? Does that change wall-clock time even when tok/s is higher?

## Step 11: Write it up
- [ ] **Do:** README with the setup, both results tables, your predictions vs what you measured, 2–3 example disagreements, and "what surprised me".

## Stretch
- [ ] Add Q8_0 and Q2_K to get an accuracy-vs-size curve. Watch the disk space.
- [ ] Compare your Q4_K_M with the official one. Are the files identical? Look up imatrix.
- [ ] Bootstrap confidence intervals for accuracy.
- [ ] Re-run llama-bench with `-ngl 0` to see CPU vs GPU.

---

## Review
<!-- Fill in when done: results, what worked, what to change -->
