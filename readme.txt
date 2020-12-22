
Dataset: 

Ekte grayscale 8-bit PNG formatında 3 sınıf  ve aynı sınıfların kusurlu olan versiyon datası paylaşıldı. 
• Hatasız görsel dosyaları: Class_1, Class_2, Class_3 (1000 image/class)
• Hatalı görsel dosyaları: Class_1_def, Class_2_def, Class_3 _def olarak belirtilmiştir. (150 image/class)

Hedef:

⁃ Image classification algoritması design ederek, görüntüleri doğru sınıflandırmak ve hatalı olup olmadıklarını tesbit etmek (output: sınıf adı ve hatlı olup olmadığı olacaktır.)
⁃ Hatalı görsellerde hatalı alanları tesbit etmek ve bounding box ile işaretlemek. Ekte örnek işaretleme gösterilmiştir.
⁃ Bonus: Bu algoritmanın web kameradan göstereceğiniz print edilmiş bir image çıktısını doğru sınıflandırması ve bounding box yaratmasını sağlayacak ek fonksiyon geliştirmek.


Buradaki yaklaşımınız ve neden yaptığınızı belirtmeniz yüksek doğruluklu sonuçtan daha önemli olacaktır. Hangi modeli neden kullandığınız, modeli nasıl evaluate ettiniz(evaluation metric olarak ne kullanırdınız vs.), vs. belirtmeniz beklenmektedir. Python veya kullandığınız başka bir dille geliştirme yapabilirsiniz. 


label.txt dosyası kusurlu alanın elipse bilgilerini vermektedir.

Column-1: [semi-major axis] 
Column-2: [semi-minor axis] 
Column-3: [rotation angle]
Column-4: [x-position of the centre of the ellipsoid] 
Column-5: [y-position of the centre of the ellipsoid] 

- The rotasyon açısı saat yönünün tersine hesaplanmıştır (pozitif açı). 
- x- and y-coordinatları: görselin orjinin yukarı-sol köşe olarak alınmıştır.




