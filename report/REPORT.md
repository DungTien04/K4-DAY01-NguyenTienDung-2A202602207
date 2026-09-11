# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:*11/9/2026*

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không 

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):   
> {468,cab,1,0.510915,ImageNet-1K}
- Record này mô tả toàn ảnh như thế nào? 
> Nó gán đúng 1 nhãn đại diện cho toàn bộ bức ảnh, không chỉ rõ 1 đối tượng cụ thể nào cả
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
> Do tập dữ liệu của ImageNet-1K định nghĩa và model yolo11n học để tự gán nhãn
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
> Đầu tiên giữ id để giúp máy tính xử lý nhanh chóng bởi nó là các index, còn giữ tên lớp để con người có thể đọc hiểu trực quan, còn taxonomy giúp xác định tập nhãn và phạm vi mà model có thể học
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
> Cần quy định rõ cho từng chủ thể ví dụ vật to nhất hay ở trung tâm bức ảnh. 
- Vì sao model score không phải ground truth?
> Vì điểm số chỉ là độ tự tin của máy tính, máy tính vẫn có thể nhầm lẫn hoặc đoán bừa với điểm số rất cao.
## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
> {bus,0.912558,[93.17,187.95,223.01,320.91],58.06,50.72}
- Diễn giải vị trí box bằng lời:
> Cái hộp nằm từ tọa độ x_min đến x_max (từ trái qua phải), từ y_min đến y_max (từ trên xuống dưới), gốc tọa độ (0,0) ở góc trên bên trái của bức ảnh.
- So sánh số prediction ở hai threshold:
> Threshold ở ngưỡng thấp khoanh và tạo nhiều bouding box hơn với cả các vật nhỏ và khó nhìn, trong khi threshold ở ngưỡng cao tạo ít box hơn nhưng bao quanh toàn bộ object to và rõ ràng.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
> Nó ảnh hưởng lớn đến khối lượng công việc của reviewer trong việc kiểm tra
- Đề xuất một quy tắc box chặt:
> Bounding box phải ôm sát các mép  ngoài cùng của vật thể không được để thừa hoặc trống quá nhiều.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
> Cần guideline quy định rõ phải khoanh toàn bộ( đoán phần bị che dựa vào tỷ lệ cân đối củ phần nhìn thấy) hay chỉ phần nhìn thấy của object.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
> {traffic-001,bus,0.925745,[
        148.0,
        189.0
      ],
      [
        147.0,
        190.0
      ],
      [
        145.0,
        190.0
      ],
      [
        143.0,
        192.0
      ],
      [
        142.0,
        192.0
      ],
      [
        141.0,
        193.0
      ],
      [
        139.0,
        193.0
      ],
      [
        137.0,
        195.0
      ],}
- Polygon bổ sung chi tiết gì so với box?
> Nó có thể khoanh vùng chính xác hình dáng của object mà không bị ảnh hưởng bởi hình chữ nhật.
- `instance_id` dùng để làm gì và không phải loại ID nào?
> Nó là ID duy nhất cho mỗi object trong ảnh phân đoạn dùng để phân biệt các object khác nhau
- Đề xuất một quy tắc biên mask:
> Polygon phải ôm sát các mép ngoài cùng của object không được để thừa hoặc trống quá nhiều.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
> Cần guideline quy định rõ phải khoanh toàn bộ( đoán phần bị che dựa vào tỷ lệ cân đối củ phần nhìn thấy) hay chỉ phần nhìn thấy của object. Nếu là vật thể nhỏ thì nên ưu tiên khoanh toàn bộ còn vật thể lớn thì nên khoanh phần nhìn thấy.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn cho ảnh | Ảnh có quá nhiều thứ  |Chọn chủ đề chính theo quy tắc  | Kiểm tra lại xem có chọn đúng chủ thể chính không  |
| Phát hiện vật thể | Bounding box cho từng object | Nhiều object bị cắt/che khuất  | Khoanh toàn bộ object nếu phần nhìn thấy còn nhiều  | Kiểm tra lại xem có khoanh toàn bộ object không và vật còn lại nhiều không |    
| Instance segmentation | Polygon cho mỗi object | Cạnh mờ không rõ ràng  | Khoanh ôm sát, không để thừa hoặc trống quá nhiều, đoán phần bị che dựa vào tỷ lệ cân đối củ phần nhìn thấy. Nếu là vật thể nhỏ thì nên ưu tiên khoanh toàn bộ còn vật thể lớn thì nên khoanh phần nhìn thấy | Kiểm tra lại xem có khoanh toàn bộ object không và vật còn lại nhiều không |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
> Không chia sẻ hình ảnh, dữ liệu, kết quả lên mạng hoặc cho người không có thẩm quyền.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
>  Nếu phát hiện dữ liệu lạ/nhạy cảm, tôi sẽ dừng lại và báo ngay cho Giảng viên/Mentor.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
