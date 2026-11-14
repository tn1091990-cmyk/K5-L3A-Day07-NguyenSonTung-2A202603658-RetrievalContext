# Báo cáo cá nhân — Day 7 Agentic RAG

## Dữ liệu và benchmark

- **Học viên:** Nguyễn Sơn Tùng — MSSV 2A202603658, lớp L3A.
- **Corpus:** 30 snapshot HTML TechPowerUp CPU Database cho AMD Ryzen; kế thừa Day 6. Mỗi HTML là một source unit `page=1`, không phải trang vật lý.
- **Quyền sử dụng:** Day 6 ghi chưa xác minh quyền tái sử dụng; dữ liệu dùng trong phạm vi bài lab.
- **Chunks:** 270 Strategy B paragraph chunks (1000 ký tự tối đa). IDs/text khớp với 42/42 chunk rows được so sánh từ trace Day 6; metadata được tạo bằng Day 6 `metadata_cpu`.
- **Vector store:** PostgreSQL 16 + pgvector, table `rag_chunks`, strategy B, cosine distance. Embedding `embeddinggemma-2`, thực tế probe xác nhận 768 chiều. Đã dựng lại và index 270 vector trong local database.
- **Benchmark:** [benchmark_day7.json](../../data/benchmark_day7.json): dev 15 answerable + 2 thiếu evidence; fresh test 5 answerable + 2 thiếu evidence. 5 câu test answerable dùng CPU pages chưa dùng làm gold trong Day 6 benchmark. Gold evidence được xác nhận trong chunk nguồn và không được truyền vào retriever hay agent.
- Câu thiếu evidence gần chủ đề: giá hiện tại tại Việt Nam (trang có thể nêu launch MSRP), bảo hành Việt Nam, số cổng USB bo mạch chủ và số màn hình 8K.

## Retrieval và thử nghiệm

Đã chạy ba phương pháp trên cùng corpus/index: Day 6 dense Strategy B, BM25 lexical, và hybrid RRF (`k=60`). Thử nghiệm dev thay đổi tham số BM25 và context budget. BM25 dùng Unicode word tokens, không tách từ tiếng Việt. Hit@1 dưới đây yêu cầu gold evidence nằm trong chunk đầu; Evidence Hit@3/context cũng yêu cầu chính xác evidence trong top 3/context. Context budget tính bằng ký tự.

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Median ms/query | API cost |
|---|---|---:|---:|---:|---:|---:|
| Dense Day 6 B, top-3, budget 2400 | dev | 13/15 | 14/15 | 14/15 | 63.13 | $0 |
| BM25 k1=1.5, b=.75, top-3, budget 2400 | dev | 11/15 | 15/15 | 14/15 | 0.64 | $0 |
| Hybrid RRF-60, top-3, budget 2400 | dev | 13/15 | 15/15 | 14/15 | 72.74 | $0 |
| BM25 k1=.8, b=.25, top-3, budget 2400 | dev | 14/15 | 15/15 | 15/15 | 0.87 | $0 |
| BM25 k1=.8, b=.25, top-3, budget 3600 — chọn trên dev | dev | 14/15 | 15/15 | 15/15 | 0.56 | $0 |
| Hybrid RRF-60, top-3, budget 1200 | dev | 13/15 | 15/15 | 13/15 | 66.71 | $0 |
| BM25 k1=1.5, b=.75, top-3, budget 2400 | fresh test | 4/5 | 5/5 | 5/5 | 0.47 | $0 |
| BM25 k1=.8, b=.25, top-3, budget 3600 — selected | fresh test | 5/5 | 5/5 | 5/5 | 0.46 | $0 |

Thời gian là median retrieval/query sau khi corpus và model sẵn sàng; dense/hybrid gồm embedding query và pgvector. Không gồm thời gian khởi động/đọc HTML hoặc điện năng. Đây là xử lý local, không phát sinh API cost.

### Giả thuyết và lỗi dev

1. **BM25 k1=.8, b=.25:** giảm độ bão hòa tần suất và ảnh hưởng chuẩn hóa độ dài sẽ ưu tiên term trùng cụ thể hơn. Hit@1 tăng 11/15 → 14/15; Evidence Hit@3 vẫn 15/15.
2. **Tăng budget lên 3600:** giữ thứ hạng BM25 đã chọn và dành đủ chỗ cho evidence. Với budget 2400, Context Evidence Rate là 15/15; tăng lên 3600 vẫn 15/15 trong lượt chạy thực tế nhưng giảm rủi ro chunk không vừa. Đối chứng budget 1200 ở hybrid giảm context evidence xuống 13/15.

Hai lỗi dev đã đọc trace. **D06 dense:** câu hỏi codename Ryzen 7 9700X trả về đúng trang nhưng top-3 không có chunk chứa “Granite Ridge”; hybrid kéo evidence vào top-3 nhưng budget 2400 bỏ chunk đó khỏi context. **D04 BM25 baseline:** trang Ryzen 5 8600G xuất hiện trong top-3, nhưng context budget 2400 bỏ đoạn Radeon 760M; cấu hình tuned với budget 3600 đưa đoạn đó vào. D13 cũng cho thấy dense xếp Ryzen 7 7700X trên Ryzen 7 7700 dù evidence có ở rank sau. Các lỗi xuất phát từ phân hạng chunk và giới hạn context.

## Đánh giá test: context và thiếu evidence

Đã đọc cả 5 context có đáp án cho baseline và cấu hình đã chọn. Context audit thủ công theo rubric 0–2: **baseline 5 câu mức 2, 0 mức 1, 0 mức 0; cấu hình đã chọn 5 mức 2, 0 mức 1, 0 mức 0**. Cả 5 context cấu hình chọn có evidence chính xác cùng source/page; T02 ở baseline chỉ có evidence sau rank 1 nhưng vẫn được đưa vào context. Hai case T06/T07 retrieve trang CPU gần chủ đề và có thể hiển thị launch MSRP, nhưng không có giá hiện tại tại Việt Nam hoặc số lượng màn hình 8K; agent cần nhận ra điều kiện còn thiếu và từ chối.

## Agent và câu trả lời

Agent dùng controller OpenAI-compatible với tools `search`, `answer` và `abstain`, model `gemma-4-E4B-it-UD-Q4_K_XL.gguf` tại local endpoint `9932`; embedding vẫn dùng `embeddinggemma-2` ở `9931`. BM25 cấu hình đã chọn (`k1=.8`, `b=.25`, top-3, context budget 3600) cung cấp context. Mỗi query gọi search một lần và dùng hai lượt model; traces giữ tool calls, kết quả, context cuối, citations, token usage và thời gian. Model được hướng dẫn coi nội dung tài liệu là dữ liệu không đáng tin cậy.

Các câu trả lời và citation đã được đọc đối chiếu thủ công với context/chunk nguồn. Sau chuẩn hóa lỗi định dạng citation và khôi phục hai câu trả lời có nội dung/citation hợp lệ nhưng model chọn nhãn abstain, agent đạt:

| Split/cấu hình | Grounded answer (0/1/2) | Mean | Citation support | Có citation | Đúng abstain | False abstain | Provider error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dev — selected BM25 | 0/0/15 trên 15 answerable | 2.00/2 | 21/21 | 15/15 | 2/2 | 0/15 | 0 |
| Test — baseline BM25 | 0/0/5 trên 5 answerable | 2.00/2 | 9/9 | 5/5 | 2/2 | 0/5 | 0 |
| Test — selected BM25 | 0/0/5 trên 5 answerable | 2.00/2 | 8/8 | 5/5 | 2/2 | 0/5 | 0 |

Các nhãn trên được ghi trong `reviews.json`; cách tổng hợp là một citation boolean cho mỗi citation và grade grounded answer 0–2 do người chấm xác nhận. Agent dùng tool search ở 17/17 dev, 7/7 baseline test và 7/7 selected test cases. Latency median mỗi query khoảng 63.3 giây dev, 53.2 giây baseline test và 73.3 giây selected test; đây là thời gian model local, không phải chi phí API. Model đôi khi gọi tool abstain cho câu trả lời thực tế có citation; trace giữ sự kiện gốc `answer_decision`, bước hậu xử lý chỉ khôi phục câu trả lời khi citation trỏ vào final context và câu trả lời không có ngôn ngữ từ chối. D17 vẫn abstain đúng nhưng phần giải thích bị cắt cụt. Xem [dev traces](../../results/agent-bm25-dev/traces.json), [dev review metrics](../../results/agent-bm25-dev/agent_metrics.json), [baseline test metrics](../../results/agent-bm25-baseline-test/agent_metrics.json), và [selected test metrics](../../results/agent-bm25-selected-test/agent_metrics.json).

## Tái lập

Từ root Day 7 với Python 3.11+ và Day 6 dependencies:

```powershell
py -3 -m src.index_day6
py -3 -m src.run_retrieval --split dev --output results/retrieval-dev
py -3 -m src.run_retrieval --split test --output results/retrieval-test
py -3 -m src.run_agent --split dev --method bm25 --bm25-k1 0.8 --bm25-b 0.25 --top-k 3 --budget-chars 3600 --output results/agent-bm25-dev
py -3 -m src.run_agent --split test --method bm25 --bm25-k1 1.5 --bm25-b 0.75 --top-k 3 --budget-chars 2400 --output results/agent-bm25-baseline-test
py -3 -m src.run_agent --split test --method bm25 --bm25-k1 0.8 --bm25-b 0.25 --top-k 3 --budget-chars 3600 --output results/agent-bm25-selected-test
```

Set `LLM_MODEL` tới ID chat/instruct do endpoint phục vụ; nếu chat chạy endpoint khác, set `LLM_BASE_URL` và `LLM_API_KEY`. Dev chạy tất cả phương pháp/thử nghiệm; test chỉ chạy baseline và cấu hình chọn. Artifacts: [dev retrieval](../../results/retrieval-dev/retrieval_summary.json), [test retrieval](../../results/retrieval-test/retrieval_summary.json), [dev agent metrics](../../results/agent-bm25-dev/agent_metrics.json), [selected test agent metrics](../../results/agent-bm25-selected-test/agent_metrics.json). Python đo: 3.15.2; embedding model 768 chiều; PostgreSQL 16/pgvector 0.8.7.

## Kết luận và giới hạn

Agent dùng Gemma 4 E4B IT qua endpoint local; review thủ công đạt 15/15 grounded dev và 5/5 selected-test, citations hỗ trợ 21/21 và 8/8, cùng 2/2 abstentions đúng trên mỗi split. Test baseline cũng đạt 5/5 grounded, citations 9/9, abstentions 2/2. Model latency cao (median 53–73 giây/query); tool-call label đôi lúc không khớp với câu trả lời, đã lưu trace nguyên bản và áp dụng recovery có điều kiện như mô tả trên.

BM25 tuned (`k1=.8`, `b=.25`) với budget 3600 được chọn trên dev và đạt 5/5 trên ba retrieval metrics của fresh test. Hybrid đạt Evidence Hit@3 15/15 trên dev, nhưng selected BM25 có Hit@1 cao hơn và đủ evidence context. Chất lượng câu trả lời/citation/abstention chưa kết luận đến khi chạy chat model thật. Corpus hẹp, một nguồn, quyền tái sử dụng chưa xác minh. Thử thách nâng cao 100 trang không thực hiện.
