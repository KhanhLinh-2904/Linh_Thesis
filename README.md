Dưới đây là file hướng dẫn chạy code được viết lại một cách rõ ràng và chi tiết hơn:

# Hướng Dẫn Chạy Code

## Link Dataset
Tải kết quả dataset từ link sau:
[Dataset](https://s.pro.vn/WTPD)
Tải dataset LowlightFace dataset từ link sau:
[LowlightFace](https://short.com.vn/pkgY)
Tải dataset SuperDarkFace dataset từ link sau:
[SuperDarkFace](https://s.pro.vn/LY61)
## Môi Trường
- Node.js: v10.19.0
- Docker: 26.1.3
- Nginx: 1.18.0

## Client 
[github]( https://github.com/KhanhLinh-2904/Thesis)
1. Cài đặt các gói cần thiết:
	```bash
	npm install
	```

2. Xây dựng dự án:
	```bash
	npm run build
	```

3. Khởi động các dịch vụ bằng Docker Compose:
	```bash
	docker-compose up
	```

4. Lấy Container ID của container bạn muốn truy cập:
	```bash
	docker ps
	```

## Server
[github]( https://github.com/KhanhLinh-2904/Thesis_Final)
1. Khởi động các dịch vụ bằng Docker Compose:
	```bash
	docker-compose up
	```

2. Truy cập vào container:
	```bash
	docker exec -it <ContainerID> bash
	```

3. Chạy script Python:
	```bash
	python test_fas.py
	```

## Cấu Hình Nginx
1. Di chuyển đến thư mục cấu hình của Nginx:
	```bash
	cd /etc/nginx/sites-enabled
	```

2. Tạo file cấu hình mới cho server AI:
	```bash
	sudo touch ai_server
	```

3. Khởi động lại Nginx để áp dụng cấu hình mới:
	```bash
	sudo systemctl restart nginx
	```

Lưu ý: Thay `<ContainerID>` bằng ID thực tế của container bạn muốn truy cập.
