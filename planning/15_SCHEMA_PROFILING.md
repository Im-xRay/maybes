# Báo cáo profiling schema bộ training

**Chế độ:** `full_stream`.
**Quyền riêng tư:** không ghi giá trị ô dữ liệu; chỉ ghi tên cột, số lượng, kiểu suy luận và nhãn heuristic.

## Tóm tắt

- Doanh nghiệp: 30
- Bảng/file worksheet: 339
- Định dạng bảng: `{"csv": 20, "sqlite": 17, "xlsx": 302}`
- Tổng dòng đã đọc: 3997076
- Bảng lỗi khi đọc: 0
- Cột được gắn ứng viên khóa (heuristic): 230
- Phân loại ngành từ README (heuristic): `{"ambiguous": 1, "construction_materials": 5, "other_or_unknown": 18, "pharmacy": 6}`

## Cột thường gặp

Tên được chuẩn hóa bỏ dấu câu/chữ hoa để thống kê tần suất; không đồng nghĩa rằng các cột có cùng tên chắc chắn cùng nghĩa.

| Tên chuẩn hóa | Số lần xuất hiện | Kiểu quan sát được | Cột duy nhất trong mẫu |
|---|---:|---|---:|
| `onvitinh` | 129 | `{"text": 129}` | 0 |
| `trangthai` | 129 | `{"text": 129}` | 0 |
| `maonhang` | 120 | `{"text": 120}` | 3 |
| `khachhang` | 95 | `{"text": 95}` | 0 |
| `soluong` | 74 | `{"integer": 18, "mixed": 56}` | 0 |
| `mahoaon` | 72 | `{"text": 72}` | 59 |
| `ongia` | 69 | `{"integer": 14, "text": 55}` | 0 |
| `thanhtien` | 69 | `{"integer": 11, "mixed": 1, "text": 57}` | 0 |
| `giamgia` | 66 | `{"integer": 66}` | 0 |
| `thoigian` | 66 | `{"date_text": 62, "datetime": 2, "mixed": 2}` | 6 |
| `tenhanghoa` | 63 | `{"text": 63}` | 5 |
| `tensanpham` | 63 | `{"text": 63}` | 4 |
| `soluongquyoi` | 62 | `{"integer": 62}` | 0 |
| `thoigiantao` | 62 | `{"datetime": 62}` | 12 |
| `tenkhachhang` | 61 | `{"text": 61}` | 6 |
| `makh` | 60 | `{"text": 60}` | 2 |
| `tongtienhang` | 59 | `{"integer": 59}` | 0 |
| `conno` | 58 | `{"integer": 58}` | 0 |
| `khachatra` | 58 | `{"integer": 58}` | 0 |
| `tongtien` | 58 | `{"integer": 39, "mixed": 19}` | 0 |
| `ngay` | 57 | `{"date_text": 57}` | 0 |
| `giasanpham` | 55 | `{"integer": 55}` | 0 |
| `ngayathang` | 55 | `{"date_text": 55}` | 0 |
| `soluongsanpham` | 55 | `{"integer": 21, "mixed": 34}` | 0 |
| `trangthaionhang` | 55 | `{"text": 55}` | 0 |
| `ghichu` | 45 | `{"empty": 37, "text": 8}` | 0 |
| `tongsaugiamgia` | 40 | `{"integer": 40}` | 0 |
| `hang` | 36 | `{"text": 36}` | 1 |
| `tien` | 36 | `{"integer": 2, "text": 34}` | 0 |
| `on` | 34 | `{"mixed": 34}` | 0 |
| `unnamed10` | 34 | `{"text": 34}` | 0 |
| `unnamed2` | 34 | `{"mixed": 17, "text": 17}` | 0 |
| `unnamed3` | 34 | `{"text": 34}` | 0 |
| `unnamed6` | 34 | `{"text": 34}` | 0 |
| `unnamed9` | 34 | `{"mixed": 34}` | 0 |
| `mahang` | 32 | `{"text": 32}` | 15 |
| `tenhang` | 23 | `{"text": 23}` | 12 |
| `vt` | 22 | `{"text": 22}` | 0 |
| `kenhban` | 20 | `{"text": 20}` | 0 |
| `idcuahang` | 17 | `{"text": 17}` | 1 |
| `maughichu` | 17 | `{"empty": 17}` | 0 |
| `unnamed4` | 17 | `{"mixed": 17}` | 0 |
| `unnamed5` | 17 | `{"text": 17}` | 0 |
| `unnamed7` | 17 | `{"text": 17}` | 0 |
| `unnamed8` | 17 | `{"text": 17}` | 0 |
| `loaihanghoa` | 16 | `{"text": 16}` | 0 |
| `baohanh` | 15 | `{"empty": 15}` | 0 |
| `mahhlienquan` | 15 | `{"empty": 15}` | 0 |
| `trongluong` | 15 | `{"empty": 15}` | 0 |
| `manhaphang` | 14 | `{"text": 14}` | 6 |
| `stt` | 14 | `{"integer": 14}` | 0 |
| `mahanghoa` | 13 | `{"text": 13}` | 7 |
| `thuonghieu` | 13 | `{"empty": 13}` | 0 |
| `uocbantructiep` | 13 | `{"integer": 13}` | 0 |
| `thuoctinh` | 12 | `{"empty": 12}` | 0 |
| `nguoitao` | 11 | `{"text": 11}` | 0 |
| `tonlonnhat` | 11 | `{"integer": 11}` | 0 |
| `vitri` | 11 | `{"empty": 11}` | 0 |
| `giaban` | 10 | `{"integer": 9, "text": 1}` | 1 |
| `giavon` | 10 | `{"integer": 6, "mixed": 1, "text": 3}` | 2 |

## Chất lượng và cảnh báo

- Cảnh báo header/kết cấu bảng: `{"blank_header_columns:2,3,4,6,7,9,10": 17, "blank_header_columns:2,3,5,6,8,9,10": 17, "leading_metadata_rows_skipped": 81, "row_2_exceeds_header_width": 34}`
- Lỗi đọc bảng: 0; xem `data/train/schema_profile.json` để biết bảng cụ thể.

## Diễn giải khóa và quan hệ

- Khóa chính SQLite được đọc từ metadata SQLite.
- Khóa ứng viên CSV/XLSX chỉ được gợi ý nếu giá trị không null trong tối đa 10.000 dòng đầu không lặp; đây không phải xác nhận khóa chính.
- Khóa ngoại chưa được xác nhận tự động. So khớp tên cột hoặc bản ghi trùng nhau có thể suy ra quan hệ sai; cần kiểm tra README, mã nghiệp vụ và tính duy nhất trước khi join.
- Quan hệ gợi ý chỉ dựa trên tên cột trùng khớp: 227 (tối đa 500 được ghi trong JSON profile); mọi quan hệ cần xác nhận bằng bản ghi khớp, tính duy nhất và rulebook.
- Cột số điện thoại/mã số có thể trông như số; không đổi mã định danh sang kiểu số để tránh mất số 0 đầu.

## Câu hỏi, đáp án và tài liệu nghiệp vụ

Thống kê JSON an toàn: `{"answers": {"error": null, "malformed_records": 0, "record_count": 3000, "records_with_alias": 167, "types": {"boolean": 23, "date": 145, "list": 193, "number": 1884, "percent": 578, "string": 88, "table": 19, "unanswerable": 70}, "unique_id_count": 3000}, "calendar": {"max_date": "2027-12-31", "min_date": "2024-01-01", "row_count": 1461}, "question_answer_id_alignment": {"extra_answer_ids": 0, "unmatched_question_ids": 0}, "questions": {"as_of_date_max": "2026-09-20", "as_of_date_min": "2025-01-15", "business_counts": {"biz_101": 40, "biz_102": 119, "biz_103": 119, "biz_104": 119, "biz_105": 119, "biz_106": 119, "biz_107": 40, "biz_108": 119, "biz_109": 118, "biz_110": 40, "biz_111": 118, "biz_112": 118, "biz_113": 40, "biz_114": 40, "biz_115": 118, "biz_116": 118, "biz_117": 118, "biz_118": 118, "biz_119": 118, "biz_120": 118, "biz_121": 118, "biz_122": 118, "biz_123": 118, "biz_124": 118, "biz_125": 40, "biz_126": 40, "biz_127": 118, "biz_128": 118, "biz_129": 118, "biz_130": 118}, "error": null, "malformed_records": 0, "overlapping_keyword_signals": {"customer": 44, "date_or_period": 1837, "debt": 67, "inventory": 15, "list_or_table": 28, "ranking": 331, "sales_or_revenue": 1022, "time_comparison": 165}, "record_count": 3000, "unique_id_count": 3000}, "rulebook_headings": ["DataMate — Rulebook (bộ train)", "1. Đơn hàng và doanh thu", "2. Thời gian", "3. Xếp hạng", "4. Công nợ", "5. Kho, nhập hàng, giá vốn", "6. Theo ngành", "Cà phê", "Nhà thuốc", "Vật liệu xây dựng", "7. Không trả lời được", "8. Cách ghi trong file", "9. Đáp án và chấm điểm", "10. Từ ngữ"]}`

Cặp câu hỏi/đáp án được nối bằng `id`; câu hỏi có `business_id` và `as_of_date`. Các tín hiệu từ khóa có thể chồng lắp nhau, không phải nhãn phân loại của ban tổ chức. Kiểu đáp án được thống kê từ nhãn training, không phải đánh giá PA chính thức.

Các nhóm nghiệp vụ và từ khóa cần tổng quát hóa: bán hàng/doanh thu, công nợ, tồn kho, nhập hàng, xếp hạng, so sánh thời gian, hàng/hạn dùng và yêu cầu thiếu dữ liệu. Phân loại ngành trong báo cáo là heuristic từ README, không phải nhãn của ban tổ chức.

## Giới hạn và hành động

- CSV thường cung cấp giá trị dạng chuỗi; kiểu nghiệp vụ được suy ra từ tên/giá trị bằng thống kê, không chuyển đổi dữ liệu nguồn.
- Bộ đọc đã nhận diện và bỏ qua các dòng metadata/tiêu đề đầu trang có cảnh báo ở 81 worksheet; cần xác nhận `header_row` trước khi sử dụng vì đây là heuristic.
- Có 34 worksheet mà dòng 2 rộng hơn dòng header; các cột header trống vẫn được giữ với tên `__unnamed_N` và cảnh báo.
- Profiling nhận diện null/kiểu/tên cột và khóa ứng viên heuristic; khóa ngoại, đơn vị tiền và ý nghĩa nghiệp vụ của cột vẫn cần kiểm tra với README/rulebook từng business.
- Query planner chỉ nên chạy trên bảng/cột đã phân biệt rõ và quan hệ có căn cứ. Chính sách abstain/unanswerable cần chốt sau khi xác minh evaluator chính thức.
- Profile đầy đủ nằm ở `data/train/schema_profile.json` (artifact cục bộ, không đưa dataset/profile vào commit). Có thể truyền `--max-rows-per-table` để chạy mẫu giới hạn khi kiểm tra nhanh.
