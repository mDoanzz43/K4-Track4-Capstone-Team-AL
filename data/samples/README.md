# Ảnh mẫu để test tab "🖼️ Tải ảnh"

Kéo-thả một file vào tab **Tải ảnh**, chọn đúng loại ảnh, rồi bấm **Trích nét → ▶ Robot vẽ**.

| Thư mục | Chọn loại | Nội dung |
|---|---|---|
| `hanzi/` | 字 Chữ / nét mực | Chữ Hán: chữ font, thư pháp, bản dập bia (chữ sáng nền tối), ảnh giả lập chụp chữ trên giấy |
| `lineart/` | 字 Chữ / nét mực | Tranh nét đơn giản (mặt mèo, ngôi nhà) |
| `portrait/` | 👤 Chân dung / ảnh chụp | Ảnh chân dung và tranh chân dung |

## Kết quả tham khảo (mức chi tiết 0.5, tốc độ 1×)

| File | Số nét | Robot vẽ ước tính | Ghi chú |
|---|---|---|---|
| `hanzi/font_yong_serif.png` 永 | 7 | ~0.3 phút | "Vĩnh tự bát pháp", chữ đơn giản nhất để thử |
| `hanzi/font_long_serif.png` 龍 | 17 | ~0.6 phút | Chữ nhiều nét |
| `hanzi/font_ai_sans_bold.png` 愛 | 13 | ~0.5 phút | Font đậm |
| `hanzi/font_fu_serif.png` 福 | 11 | ~0.5 phút | |
| `hanzi/font_shufa_serif.png` 書法 | 15 | ~0.4 phút | Hai chữ |
| `hanzi/photo_like_fu_on_paper.jpg` 福 | 12 | ~0.4 phút | Giả lập ảnh chụp: giấy ngà, ánh sáng lệch, phối cảnh nghiêng, mép bàn tối |
| `hanzi/dao_calligraphy.png` 道 | 9 | ~0.4 phút | Thư pháp |
| `hanzi/biang_regular_script.png` 𰻞 (biáng) | 49 | ~0.9 phút | Một trong những chữ Hán nhiều nét nhất (khoảng 56–58 nét khi viết tay) |
| `hanzi/ouyang_xun_kaishu_rubbing.jpg` | 134 | ~1.2 phút | Bản dập bia Cửu Thành Cung của Âu Dương Tuân: chữ sáng trên nền đá tối, cả trang 24 chữ |
| `hanzi/yao_mengqi_jiuchenggong_1883.png` | 135 | ~1.3 phút | Bản lâm mô cùng văn bản, mực đen trên nền trắng |
| `lineart/cat_face.png` | 12 | ~0.4 phút | |
| `lineart/house.png` | 9 | ~0.5 phút | |
| `portrait/lincoln_1863.jpg` | 245 | ~2.3 phút | Nhiều đổ bóng; tách nền tốt |
| `portrait/einstein_1947.jpg` | 132 | ~1.4 phút | Tóc bù xù, nền xám |
| `portrait/mona_lisa.jpg` | 161 | ~1.7 phút | Tranh sơn dầu |
| `portrait/van_gogh_self_portrait_1887.jpg` | 157 | ~1.5 phút | Tranh nét cọ, nhiều nét vụn |
| `portrait/marie_curie_1920.jpg` | 203 | ~1.8 phút | Áo tối trên nền tối: tách nền không đáng tin nên tự chuyển sang vẽ cả khung (nền cũng được gạch bóng) |

Muốn robot vẽ nhanh hơn: giảm **Mức chi tiết** hoặc tắt **Đổ bóng bằng nét gạch**, và kéo **Tốc độ** lên 10×.

## Nguồn và giấy phép

Các ảnh từ Wikimedia Commons đều thuộc **phạm vi công cộng (public domain)**. Chúng được tải bản thu nhỏ
(rộng 1280 px hoặc 1024 px) và lưu lại bằng OpenCV. Ảnh SVG được render sang PNG trên nền trắng.

| File | Tác giả / nguồn gốc | Năm | Trang gốc |
|---|---|---|---|
| `portrait/lincoln_1863.jpg` | Alexander Gardner | 1863 | [Commons](https://commons.wikimedia.org/wiki/File:Abraham_Lincoln_O-77_matte_collodion_print.jpg) |
| `portrait/einstein_1947.jpg` | Orren Jack Turner (bản quyền 1947 không gia hạn) | 1947 | [Commons](https://commons.wikimedia.org/wiki/File:Albert_Einstein_Head.jpg) |
| `portrait/marie_curie_1920.jpg` | Henri Manuel | ~1920 | [Commons](https://commons.wikimedia.org/wiki/File:Marie_Curie_c1920.jpg) |
| `portrait/mona_lisa.jpg` | Leonardo da Vinci (ảnh C2RMF) | 1503–1506 | [Commons](https://commons.wikimedia.org/wiki/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg) |
| `portrait/van_gogh_self_portrait_1887.jpg` | Vincent van Gogh (Google Art Project) | 1887 | [Commons](https://commons.wikimedia.org/wiki/File:Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project_(454045).jpg) |
| `hanzi/ouyang_xun_kaishu_rubbing.jpg` | Âu Dương Tuân (歐陽詢), bản dập | 632 | [Commons](https://commons.wikimedia.org/wiki/File:KaishuOuyangxun.jpg) |
| `hanzi/yao_mengqi_jiuchenggong_1883.png` | Diêu Mạnh Khởi (姚孟起) lâm mô Âu Dương Tuân | 1883 | [Commons](https://commons.wikimedia.org/wiki/File:%E5%A7%9A%E5%AD%9F%E8%B5%B7%E4%B9%9D%E6%88%90%E5%AE%AE.png) |
| `hanzi/biang_regular_script.png` | FanNihongo | 2021 | [Commons](https://commons.wikimedia.org/wiki/File:Bi%C3%A1ng_(regular_script).svg) |
| `hanzi/dao_calligraphy.png` | Nyo | — | [Commons](https://commons.wikimedia.org/wiki/File:Tao.svg) |

Các file `hanzi/font_*.png` và `hanzi/photo_like_fu_on_paper.jpg` được tự tạo bằng font **Noto Serif/Sans CJK**
(SIL Open Font License; glyph render ra ảnh được dùng tự do). `lineart/*.png` được vẽ bằng OpenCV, không có bản quyền bên thứ ba.
