import json
from datetime import date, timedelta

import pytest

from datamate.plan_serialization import (
    QueryPlanFormatError,
    query_plan_from_json,
)
from datamate.question_understanding import (
    JsonPlanAdapter,
    QuestionContext,
    QuestionUnderstandingError,
    build_planning_prompt,
)


TABLE = "sales.csv::table"
PLAN = {
    "base_table": TABLE,
    "aggregations": [
        {
            "function": "sum",
            "column": {"table": TABLE, "column": "amount"},
            "alias": "total",
        }
    ],
    "output_type": "number",
}


def test_query_plan_json_decodes_supported_nested_fields() -> None:
    payload = {
        **PLAN,
        "filters": [
            {
                "column": {"table": TABLE, "column": "amount"},
                "operator": "between",
                "value": [10, 25],
            }
        ],
        "time_range": {
            "column": {"table": TABLE, "column": "sold_at"},
            "start": "2026-01-01",
            "end": "2026-01-31",
            "inclusive_end": False,
        },
        "aggregations": [
            {
                **PLAN["aggregations"][0],
                "unit_conversion": {
                    "source_unit": {"table": TABLE, "column": "unit"},
                    "base_unit": {"table": TABLE, "column": "base_unit"},
                    "large_unit": {"table": TABLE, "column": "large_unit"},
                    "factor": {"table": TABLE, "column": "factor"},
                    "target_unit": "kg",
                },
                "time_range": {
                    "column": {"table": TABLE, "column": "sold_at"},
                    "start": "2025-12-01",
                    "end": "2025-12-31",
                },
                "filters": [
                    {
                        "column": {"table": TABLE, "column": "status"},
                        "operator": "eq",
                        "value": "completed",
                    }
                ],
            }
        ],
        "calculations": [
            {
                "function": "ratio",
                "left": "total",
                "right": "total",
                "alias": "average",
                "rounding_places": 0,
            }
        ],
        "time_buckets": [
            {
                "column": {"table": TABLE, "column": "sold_at"},
                "granularity": "month",
                "alias": "month",
            }
        ],
        "group_filters": [{"alias": "total", "operator": "gte", "value": 20}],
        "group_exclusions": [
            {
                "key": {"table": TABLE, "column": "order_id"},
                "filters": [
                    {
                        "column": {"table": TABLE, "column": "quantity"},
                        "operator": "lt",
                        "value": 0,
                    }
                ],
            }
        ],
        "count_groups": True,
        "metadata": {"reason": "monthly total"},
    }

    plan = query_plan_from_json(json.dumps(payload))

    assert plan.filters[0].value == (10, 25)
    assert plan.time_range is not None
    assert plan.time_range.inclusive_end is False
    aggregation = plan.aggregations[0]
    assert aggregation.time_range is not None
    assert aggregation.time_range.start == "2025-12-01"
    assert aggregation.filters[0].value == "completed"
    assert aggregation.unit_conversion is not None
    assert aggregation.unit_conversion.target_unit == "kg"
    assert plan.calculations[0].rounding_places == 0
    assert plan.time_buckets[0].granularity == "month"
    assert plan.group_filters[0].alias == "total"
    assert plan.group_exclusions[0].key.column == "order_id"
    assert plan.group_exclusions[0].filters[0].operator == "lt"
    assert plan.count_groups is True
    assert plan.metadata["reason"] == "monthly total"


def test_query_plan_json_accepts_product_specific_base_unit_conversion() -> None:
    payload = {
        **PLAN,
        "aggregations": [
            {
                **PLAN["aggregations"][0],
                "unit_conversion": {
                    "source_unit": {"table": TABLE, "column": "unit"},
                    "base_unit": {"table": TABLE, "column": "base_unit"},
                    "large_unit": {"table": TABLE, "column": "large_unit"},
                    "factor": {"table": TABLE, "column": "factor"},
                    "target_unit": None,
                },
            }
        ],
    }

    plan = query_plan_from_json(json.dumps(payload))

    assert plan.aggregations[0].unit_conversion is not None
    assert plan.aggregations[0].unit_conversion.target_unit is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"base_table":"sales.csv::table","unexpected":true}',
        '{"base_table":"sales.csv::table","limit":true}',
        '{"base_table":"sales.csv::table","base_table":"other::table"}',
        '{"base_table":"sales.csv::table","filters":[{"column":{"table":"sales.csv::table","column":"amount"},"operator":2,"value":10}]}',
        '{"base_table":"sales.csv::table","value":NaN}',
    ],
)
def test_query_plan_json_rejects_malformed_or_unsafe_shapes(raw: str) -> None:
    with pytest.raises(QueryPlanFormatError):
        query_plan_from_json(raw)


def test_query_plan_json_rejects_excessive_nesting() -> None:
    raw = "[" * 20_000 + "0" + "]" * 20_000

    with pytest.raises(QueryPlanFormatError, match="too deeply nested"):
        query_plan_from_json(raw)


def test_query_plan_json_accepts_standalone_json_code_fences() -> None:
    raw = "```json\n" + json.dumps(PLAN) + "\n```"

    assert query_plan_from_json(raw).base_table == TABLE


def test_query_plan_json_does_not_extract_json_from_thinking_or_prose() -> None:
    for raw in (
        "<think>reasoning</think>\n" + json.dumps(PLAN),
        "Here is the plan:\n" + json.dumps(PLAN),
        "```json\n" + json.dumps(PLAN) + "\n```\nextra text",
    ):
        with pytest.raises(QueryPlanFormatError):
            query_plan_from_json(raw)


def test_planning_prompt_carries_schema_and_reference_date() -> None:
    prompt = build_planning_prompt(
        "doanh thu tháng này",
        QuestionContext(
            schema={TABLE: ("amount", "sold_at")},
            as_of_date="2026-05-25",
            business_rules=("Hủy đơn không tính doanh thu.",),
            rulebook="# Nguyên tắc",
            business_readme="amount tính bằng VND",
            calendar_columns=("solar_date", "lunar_date"),
            calendar_rows=(
                {"solar_date": "2026-05-25", "lunar_date": "2026-04-09"},
            ),
        ),
    )

    assert "doanh thu tháng này" in prompt
    assert "2026-05-25" in prompt
    assert "sold_at" in prompt
    assert "Hủy đơn không tính doanh thu." in prompt
    assert "# Nguyên tắc" in prompt
    assert "amount tính bằng VND" in prompt
    assert "2026-04-09" in prompt
    assert "business_rules luôn là []" in prompt
    assert "group_exclusions khi phải loại cả nhóm" in prompt
    assert "count_distinct" in prompt
    assert "unit_conversion" in prompt
    assert "ranking giảm dần" in prompt.casefold()
    assert "time_buckets" in prompt
    assert "partial: explicit dates/months" in prompt
    assert "filters, group_by, group_filters" in prompt
    assert "Không dùng object thay cho mảng" in prompt
    assert "sum|count|count_distinct|avg|min|max" in prompt
    assert "Dùng đúng ID bảng đầy đủ" in prompt


def test_planning_prompt_selects_rulebook_sections_relevant_to_question() -> None:
    rulebook = (
        "Mặc định dùng các quy tắc trong tài liệu.\n"
        "## Công nợ\nĐơn ghi nợ và tuổi nợ tính từ ngày đặt.\n"
        "## Kho hàng\nTồn kho cuối ngày và giá vốn theo quy tắc riêng.\n"
        "## Quy tắc cho mọi câu hỏi\nNếu dữ liệu không đủ thì không suy đoán.\n"
        "## Phụ lục không liên quan\n"
        + ("Thuật ngữ kỹ thuật không liên quan đến câu hỏi. " * 300)
    )
    prompt = build_planning_prompt(
        "Tuổi nợ của khách hàng là bao nhiêu?",
        QuestionContext(
            schema={"orders.csv::orders": ("order_id", "due_date")},
            rulebook=rulebook,
        ),
    )
    payload_text = prompt.split("Ngữ cảnh JSON:\n", 1)[1].split(
        "\nQueryPlan JSON:\n", 1
    )[0]
    excerpt = json.loads(payload_text)["rulebook"]

    assert "## Công nợ" in excerpt
    assert "## Kho hàng" not in excerpt
    assert "các mục khác đã lược bỏ" in excerpt


def test_planning_prompt_avoids_unrelated_rulebook_topics() -> None:
    rulebook = (
        "Quy tắc áp dụng theo nội dung câu hỏi.\n"
        "## 1. Đơn hàng và doanh thu\nQuy định đơn hàng hợp lệ.\n"
        "## 2. Thời gian\nTuần bắt đầu vào thứ Hai.\n"
        "## 3. Xếp hạng\nQuy tắc xếp hạng và đơn vị sản phẩm.\n"
        "## 5. Kho, nhập hàng, giá vốn\nQuy tắc tồn kho và giá vốn.\n"
        "### Nhà thuốc\nQuy tắc riêng cho thuốc.\n"
        "### Vật liệu xây dựng\nQuy tắc riêng cho vật liệu.\n"
        "## 7. Không trả lời được\nKhông suy đoán khi thiếu dữ liệu.\n"
        "## 8. Cách ghi trong file\nTên bảng và cột theo dữ liệu nguồn.\n"
        "## 9. Đáp án và chấm điểm\nQuy định định dạng đáp án.\n"
        "## 10. Từ ngữ\nĐơn vị sản phẩm.\n"
        "## Phụ lục\n"
        + ("Thuật ngữ kỹ thuật không liên quan. " * 300)
    )
    prompt = build_planning_prompt(
        "Tháng 5/2026 bán ra bao nhiêu cây Ống PVC Bình Minh D50 cây 4m?",
        QuestionContext(
            schema={"sales.csv::sales": ("product", "quantity", "sold_at")},
            business_readme="Loại hình: vật liệu xây dựng",
            rulebook=rulebook,
        ),
    )
    payload_text = prompt.split("Ngữ cảnh JSON:\n", 1)[1].split(
        "\nQueryPlan JSON:\n", 1
    )[0]
    excerpt = json.loads(payload_text)["rulebook"]

    assert "## 1. Đơn hàng và doanh thu" in excerpt
    assert "## 2. Thời gian" in excerpt
    assert "## 3. Xếp hạng" in excerpt
    assert "### Vật liệu xây dựng" in excerpt
    assert "## 7. Không trả lời được" in excerpt
    assert "## 8. Cách ghi trong file" in excerpt
    assert "## 10. Từ ngữ" in excerpt
    assert "## 5. Kho, nhập hàng, giá vốn" not in excerpt
    assert "### Nhà thuốc" not in excerpt
    assert "## 9. Đáp án và chấm điểm" not in excerpt


def test_planning_prompt_bounds_large_calendar_to_relevant_rows() -> None:
    first_day = date(2024, 1, 1)
    rows = []
    for offset in range(1_461):
        day = first_day + timedelta(days=offset)
        rows.append(
            {
                "date": day.isoformat(),
                "weekday": str(day.weekday()),
                "lunar_day": str(day.day),
                "lunar_month": str(day.month),
                "lunar_year": str(day.year),
                "lunar_leap": "false",
                "holiday": "Holiday" if day.day == 1 else "",
            }
        )
    prompt = build_planning_prompt(
        "Doanh thu ngày 2026-01-12?",
        QuestionContext(
            schema={TABLE: ("amount",)},
            as_of_date="2026-01-15",
            calendar_columns=(
                "date",
                "weekday",
                "lunar_day",
                "lunar_month",
                "lunar_year",
                "lunar_leap",
                "holiday",
            ),
            calendar_rows=tuple(rows),
        ),
    )

    payload_text = prompt.split("Ngữ cảnh JSON:\n", 1)[1].split(
        "\nQueryPlan JSON:\n", 1
    )[0]
    payload = json.loads(payload_text)
    calendar = payload["calendar"]
    included_dates = {row[0] for row in calendar["rows"]}
    original_calendar_size = len(
        json.dumps(
            {
                "columns": [
                    "date",
                    "weekday",
                    "lunar_day",
                    "lunar_month",
                    "lunar_year",
                    "lunar_leap",
                    "holiday",
                ],
                "rows": [dict(row) for row in rows],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )

    assert original_calendar_size > 100_000
    assert len(json.dumps(calendar, ensure_ascii=False, separators=(",", ":"))) <= 8_192
    assert len(calendar["rows"]) < 100
    assert "2026-01-12" in included_dates
    assert "2024-06-01" in included_dates
    assert "2024-01-02" not in included_dates
    assert "missing date is unavailable" in calendar["coverage"]


def test_planning_prompt_fits_token_budget_without_dropping_required_context() -> None:
    context = QuestionContext(
        schema={TABLE: ("amount", "sold_at")},
        as_of_date="2026-01-15",
        business_rules=("Canceled transactions are excluded.",),
        rulebook=(
            "## Doanh thu\n"
            + "Doanh thu chỉ tính giao dịch hợp lệ. " * 120
            + "\n## Nội dung khác\n"
            + "Quy tắc không liên quan. " * 120
        ),
        business_readme=(
            "## Doanh thu\n"
            + "Cột amount là giá trị giao dịch. " * 120
        ),
        calendar_columns=("date", "holiday"),
        calendar_rows=tuple(
            {"date": f"2026-01-{day:02d}", "holiday": ""}
            for day in range(1, 32)
        ),
    )
    count_tokens = lambda prompt: len(prompt.split())
    prompt = build_planning_prompt(
        "Doanh thu ngày 2026-01-12?",
        context,
        max_prompt_tokens=600,
        token_counter=count_tokens,
    )
    payload_text = prompt.split("Ngữ cảnh JSON:\n", 1)[1].split(
        "\nQueryPlan JSON:\n", 1
    )[0]
    payload = json.loads(payload_text)

    assert count_tokens(prompt) <= 600
    assert payload["question"] == "Doanh thu ngày 2026-01-12?"
    assert payload["schema"] == {TABLE: ["amount", "sold_at"]}
    assert payload["business_rules"] == ["Canceled transactions are excluded."]
    assert "excerpt" in payload["rulebook"] or "omitted" in payload["rulebook"]
    assert "omitted" in payload["business_readme"]
    assert "partial" in payload["calendar"]["coverage"]


def test_planning_prompt_rejects_budget_that_cannot_hold_required_schema() -> None:
    context = QuestionContext(
        schema={TABLE: tuple(f"column_{index}" for index in range(200))},
    )

    with pytest.raises(QuestionUnderstandingError, match="Essential question/schema"):
        build_planning_prompt(
            "Tổng doanh thu?",
            context,
            max_prompt_tokens=20,
            token_counter=lambda prompt: len(prompt.split()),
        )


def test_json_plan_adapter_uses_backend_token_budget_before_generation() -> None:
    class TokenBudgetBackend:
        max_prompt_tokens = 600

        def __init__(self) -> None:
            self.prompt_tokens = None

        def count_prompt_tokens(self, prompt: str) -> int:
            return len(prompt.split())

        def generate(self, prompt: str) -> str:
            self.prompt_tokens = self.count_prompt_tokens(prompt)
            return json.dumps(PLAN)

    backend = TokenBudgetBackend()
    context = QuestionContext(
        schema={TABLE: ("amount", "sold_at")},
        rulebook=("## Sales\n" + "Revenue rule. " * 200),
        business_readme=("## Sales\n" + "Amount description. " * 200),
    )

    plan = JsonPlanAdapter(backend, model_name="local-test").create_plan(
        "Tổng doanh thu?",
        context,
    )

    assert plan.base_table == TABLE
    assert backend.prompt_tokens is not None
    assert backend.prompt_tokens <= backend.max_prompt_tokens


def test_json_plan_adapter_validates_against_discovered_schema() -> None:
    prompts: list[str] = []
    adapter = JsonPlanAdapter(
        lambda prompt: prompts.append(prompt) or json.dumps(PLAN),
        model_name="local-test",
    )

    plan = adapter.create_plan(
        "Tổng doanh thu?",
        QuestionContext(schema={TABLE: ("amount",)}),
    )

    assert plan.base_table == TABLE
    assert prompts and "Tổng doanh thu?" in prompts[0]


def test_json_plan_adapter_retries_once_after_invalid_model_json() -> None:
    prompts: list[str] = []
    outputs = iter(("not JSON", json.dumps(PLAN)))
    adapter = JsonPlanAdapter(
        lambda prompt: prompts.append(prompt) or next(outputs),
        model_name="local-test",
    )

    plan = adapter.create_plan(
        "Tổng doanh thu?",
        QuestionContext(schema={TABLE: ("amount",)}),
    )

    assert plan.base_table == TABLE
    assert len(prompts) == 2
    assert "strict JSON parser and QueryPlan validator" in prompts[1]
    assert "not JSON" not in prompts[1]


def test_json_plan_adapter_accepts_count_distinct_for_repeated_order_lines() -> None:
    plan_data = {
        "base_table": TABLE,
        "aggregations": [
            {
                "function": "count_distinct",
                "column": {"table": TABLE, "column": "order_id"},
                "alias": "order_count",
            }
        ],
        "output_type": "number",
    }

    plan = JsonPlanAdapter(
        lambda _: json.dumps(plan_data),
        model_name="local-test",
    ).create_plan(
        "Có bao nhiêu đơn?",
        QuestionContext(schema={TABLE: ("order_id", "amount")}),
    )

    assert plan.aggregations[0].function == "count_distinct"


def test_json_plan_adapter_rejects_hallucinated_columns_and_non_json() -> None:
    hallucinated = {
        **PLAN,
        "aggregations": [
            {
                "function": "sum",
                "column": {"table": TABLE, "column": "missing"},
                "alias": "total",
            }
        ],
    }
    context = QuestionContext(schema={TABLE: ("amount",)})

    with pytest.raises(QuestionUnderstandingError, match="unknown column"):
        JsonPlanAdapter(
            lambda _: json.dumps(hallucinated),
            model_name="local-test",
        ).create_plan("Tổng?", context)
    with pytest.raises(QuestionUnderstandingError, match="valid JSON"):
        JsonPlanAdapter(
            lambda _: "not JSON",
            model_name="local-test",
        ).create_plan("Tổng?", context)


@pytest.mark.parametrize(
    ("plan_data", "issue"),
    [
        ({"aggregations": PLAN["aggregations"], "output_type": "number"}, "base_table"),
        ({**PLAN, "selected_columns": "amount"}, "selected_columns must be a JSON array"),
        (
            {
                **PLAN,
                "aggregations": [
                    {
                        "function": "median",
                        "column": {"table": TABLE, "column": "amount"},
                        "alias": "median",
                    }
                ],
            },
            "unsupported aggregation",
        ),
    ],
)
def test_json_plan_adapter_rejects_missing_wrong_typed_and_unsupported_plans(
    plan_data: dict[str, object],
    issue: str,
) -> None:
    with pytest.raises(QuestionUnderstandingError, match=issue):
        JsonPlanAdapter(
            lambda _: json.dumps(plan_data),
            model_name="local-test",
        ).create_plan(
            "Tổng doanh thu?",
            QuestionContext(schema={TABLE: ("amount", "sold_at", "status")}),
        )


def test_json_plan_adapter_validates_vietnamese_question_with_multiple_conditions() -> None:
    multi_condition_plan = {
        **PLAN,
        "filters": [
            {
                "column": {"table": TABLE, "column": "status"},
                "operator": "eq",
                "value": "delivered",
            },
            {
                "column": {"table": TABLE, "column": "amount"},
                "operator": "gt",
                "value": 0,
            },
        ],
        "time_range": {
            "column": {"table": TABLE, "column": "sold_at"},
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
    }

    plan = JsonPlanAdapter(
        lambda _: json.dumps(multi_condition_plan, ensure_ascii=False),
        model_name="local-test",
    ).create_plan(
        "Tổng doanh thu đơn đã giao trong tháng 1, bỏ số tiền không dương?",
        QuestionContext(
            schema={TABLE: ("amount", "sold_at", "status")},
            as_of_date="2026-02-01",
        ),
    )

    assert len(plan.filters) == 2
    assert plan.filters[0].value == "delivered"
    assert plan.filters[1].operator == "gt"
    assert plan.time_range is not None
    assert plan.time_range.start == "2026-01-01"


def test_json_plan_adapter_rejects_unexecutable_business_rule_tags() -> None:
    tagged_plan = {
        **PLAN,
        "business_rules": ["exclude cancelled orders"],
    }

    with pytest.raises(QuestionUnderstandingError, match="business_rules are not executable"):
        JsonPlanAdapter(
            lambda _: json.dumps(tagged_plan),
            model_name="local-test",
        ).create_plan("Tổng doanh thu?", QuestionContext(schema={TABLE: ("amount",)}))


def test_json_plan_adapter_accepts_only_explicit_non_answer_statuses() -> None:
    context = QuestionContext(schema={TABLE: ("amount",)})
    ambiguous_plan = {
        "base_table": TABLE,
        "output_type": "unanswerable",
        "metadata": {
            "status": "ambiguous",
            "reason": "Chưa rõ cần tính doanh thu hay số đơn.",
        },
    }

    plan = JsonPlanAdapter(
        lambda _: json.dumps(ambiguous_plan),
        model_name="local-test",
    ).create_plan("Câu hỏi mơ hồ", context)

    assert plan.output_type == "unanswerable"
    assert plan.metadata["status"] == "ambiguous"
    no_table_abstention = {
        "base_table": None,
        "output_type": "abstain",
        "metadata": {"status": "abstained", "reason": "Không thể xác định."},
    }
    no_table_plan = JsonPlanAdapter(
        lambda _: json.dumps(no_table_abstention),
        model_name="local-test",
    ).create_plan("Câu hỏi không thuộc nghiệp vụ", context)
    assert no_table_plan.base_table is None

    insufficient_data_plan = {
        "base_table": TABLE,
        "output_type": "unanswerable",
        "metadata": {
            "status": "insufficient_data",
            "reason": "Không có cột ngày để xác định khoảng thời gian.",
        },
    }
    insufficient_data = JsonPlanAdapter(
        lambda _: json.dumps(insufficient_data_plan, ensure_ascii=False),
        model_name="local-test",
    ).create_plan(
        "Doanh thu năm 2025?",
        QuestionContext(schema={TABLE: ("amount",)}),
    )
    assert insufficient_data.output_type == "unanswerable"
    assert insufficient_data.metadata["status"] == "insufficient_data"

    ambiguous_plan["metadata"]["status"] = "no_data"
    with pytest.raises(QuestionUnderstandingError, match="metadata.status"):
        JsonPlanAdapter(
            lambda _: json.dumps(ambiguous_plan),
            model_name="local-test",
        ).create_plan("Câu hỏi mơ hồ", context)
