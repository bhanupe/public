# Part 2
[capstone_hs_recommendation.py](capstone_hs_recommendation.py)
```python
# Develop a simple recommendation engine to suggest places of interest to tourists.
# Use collaborative filtering approach with the Surprise library.

import pandas as pd
from surprise import Dataset, Reader, SVD
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from surprise.model_selection import cross_validate
import os
from datetime import datetime


def get_datetime():
    now = datetime.now()

    # It is also possible to format the output:
    return now.strftime("%Y%m%d%H%M%S")


def create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)


show = True
folder = get_datetime()
create_folder(folder)


def save_show(plot, name):
    plot.savefig(f'{folder}/{name}.png')
    if show:
        plot.show()


warnings.filterwarnings("ignore")

# part 2: Method 1
ratings_df = pd.read_csv('data/part2/tourism_rating.csv')

# Load data into Surprise's Dataset format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['User_Id', 'Place_Id', 'Place_Ratings']], reader)

# 3. Train a collaborative filtering model (SVD)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)


# Generate recommendations for a user
def get_recommendations(user_id, max_recommendations=5):
    # Get a list of all item IDs
    all_item_ids = ratings_df['Place_Id'].unique()

    # Get items the user has already rated
    rated_items = ratings_df[ratings_df['User_Id'] == user_id]['Place_Id'].unique()

    # Predict ratings for unrated items
    unrated_items = [item for item in all_item_ids if item not in rated_items]
    predictions = [algo.predict(user_id, item_id) for item_id in unrated_items]

    # Sort predictions by estimated rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get top N recommendations
    top_mr = predictions[:max_recommendations]

    return [(pred.iid, pred.est) for pred in top_mr]


# Load tourism data
tourism_df = pd.read_excel('data/part2/tourism_with_id.xlsx')

# Example: Get recommendations for users from users.csv
users_df = pd.read_csv('data/part2/user.csv')
for user_id_to_recommend in users_df['User_Id']:
    recommendations = get_recommendations(user_id_to_recommend)

    print(f"\n**Top Recommendations for User {user_id_to_recommend}**")
    for item, estimated_rating in recommendations:
        print(f"- Recommended Category: {item}, Estimated Rating: {estimated_rating:.2f}")

        try:
            # Find places in the recommended category with a similar rating
            similar_places = tourism_df[
                (tourism_df['Place_Id'] == item) &
                (tourism_df['Rating'] >= estimated_rating * 0.9) &
                (tourism_df['Rating'] <= estimated_rating * 1.1)
                ]

            print(f"- Places in category '{item}' with similar rating ---")
            if not similar_places.empty:
                for _, row in similar_places.iterrows():
                    print(
                        f"- Place: {row['Place_Id']}\n- Place: {row['Place_Name']}\n- Description: {row['Description']}\n- City: {row['City']}\n- Price: {row['Price']}\n- Rating: {row['Rating']:.2f}\n")
            else:
                print("- - No similar places found\n")
        except KeyError as e:
            print(f"- An error occurred while finding similar places. A required column is missing: {e}")

# part 2: Method 2
users = pd.read_csv("data/part2/user.csv")
places = pd.read_excel("data/part2/tourism_with_id.xlsx")
ratings = pd.read_csv("data/part2/tourism_rating.csv")

users.drop_duplicates(inplace=True)
places.drop_duplicates(inplace=True)
ratings.drop_duplicates(inplace=True)

print("sums the NAN values column-wise, giving you the total number of missing values in each column of the DataFrame")
print(users.isnull().sum())
print(places.isnull().sum())
print(ratings.isnull().sum())

sns.histplot(users['Age'], bins=20, kde=True)
plt.title("Age Distribution of Tourists")
save_show(plt, "AgeDistributionOfTourists")

sns.countplot(y='Location', data=users, order=users['Location'].value_counts().index[:10])
plt.title("Top Tourist Origins")
save_show(plt, "TopTouristOrigins")

merged = ratings.merge(users, on='User_Id').merge(places, on='Place_Id')

top_places = merged.groupby('Place_Name')['Place_Ratings'].mean().sort_values(ascending=False)
print(top_places.head())

best_cities = merged.groupby('City')['Place_Ratings'].mean().sort_values(ascending=False)
print(best_cities.head())

best_categories = merged.groupby('Category')['Place_Ratings'].mean().sort_values(ascending=False)
print(best_categories.head())

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['User_Id', 'Place_Id', 'Place_Ratings']], reader)

algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset = data.build_full_trainset()
algo.fit(trainset)

place_ids = ratings['Place_Id'].unique()

# user_id = 'U1003'
# recommendations = [(pid, algo.predict(user_id, pid).est) for pid in place_ids]
# recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
#
# print("Top Recommendations for User:", user_id)
# for pid, score in recommendations[:5]:
#     name = places[places['Place_Id'] == pid]['Place_Name'].values[0]
#     print(f"{name} (Predicted Rating: {score:.2f})")

users_df = pd.read_csv('data/part2/user.csv')
for user_id_to_recommend in users_df['User_Id']:
    recommendations = [(pid, algo.predict(user_id_to_recommend, pid).est) for pid in place_ids]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    print("\n**Top 5 Recommendations for User:**", user_id_to_recommend)
    for pid, score in recommendations[:5]:
        name = places[places['Place_Id'] == pid]['Place_Name'].values[0]
        print(f"- {name} (Predicted Rating: {score:.2f})")
```

## Method 1
Python3.12/bin/python PycharmProjects/public/analytical_processors/historcal_structures/capstone_hs_recommendation.py 

**Top Recommendations for User 1**
- Recommended Category: 139, Estimated Rating: 4.25
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 4.16
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 112, Estimated Rating: 4.08
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 4.06
- Places in category '401' with similar rating ---
- Place: 401
- Place: Taman Keputran
- Description: Ntah, mengapa nama taman ini disebut dengan taman keputran, namun jika dikaitkan dengan lokasi taman, berada di persimpangan jalan Keputran Surabaya. Selain itu, di seberang sungai terdapat juga sebuah taman yang bernama Taman Lalu Lintas Surabaya. Keunikan Taman Keputran dibandingkan dengan taman lainnya adalah adanya lampu yang kerangka tiangnya mirip dengan manusia, dengan gaya pose yang berbeda-beda itu, membuat taman ini terlihat ada sentuhan kreatif dan inovatif. Lampu-lampu itu terang dan indah pada malam hari. Untuknya, taman ini cukup representatif sebagai pilihan mendapatkan tempat tongkrongan yang tenang pada malam hari. Tak hanya itu, jalur pijat refleksi kaki juga tersedia disana. Cocok untuk pengunjung yang ingin mencoba pengobatan refleksi kaki sederhana. Melalui pemahaman tentang jalur refleksi secara mendalam, sebagian orang memilih sarana ini dalam usaha menyembuhkan penyakit._x000D_
- City: Surabaya
- Price: 0
- Rating: 4.30

- Recommended Category: 52, Estimated Rating: 3.99
- Places in category '52' with similar rating ---
- - No similar places found


**Top Recommendations for User 2**
- Recommended Category: 416, Estimated Rating: 4.06
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 146, Estimated Rating: 4.00
- Places in category '146' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.94
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.92
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 196, Estimated Rating: 3.88
- Places in category '196' with similar rating ---
- - No similar places found


**Top Recommendations for User 3**
- Recommended Category: 416, Estimated Rating: 4.11
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 314, Estimated Rating: 4.11
- Places in category '314' with similar rating ---
- Place: 314
- Place: Tafso Barn
- Description: Nama Punclut mungkin sudah cukup akrab di telinga wisatawan. Kawasan dataran tinggi di Bandung yang belakangan populer sebagai destinasi wisata. Selain menawarkan pemandangan yang indah, kawasan ini juga memiliki banyak tempat wisata untuk rekreasi. Salah satunya adalah kawasan wisata sekaligus restoran Tafso dan Boda Barn atau sering dikenal juga dengan Tafso Barn. Tempat rekreasi ini memadukan tempat makan dengan nuansa taman dan gudang yang unik. Pemandangan perbukitan yang dipotong garis horison menjadi incaran utama pengunjung. Sangat tepat nongkrong dan bersantai ketika berlibur di Bandung.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 254, Estimated Rating: 4.08
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 44, Estimated Rating: 4.07
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 399, Estimated Rating: 4.04
- Places in category '399' with similar rating ---
- - No similar places found


**Top Recommendations for User 4**
- Recommended Category: 322, Estimated Rating: 4.31
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 416, Estimated Rating: 4.20
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 79, Estimated Rating: 4.08
- Places in category '79' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 4.05
- Places in category '321' with similar rating ---
- Place: 321
- Place: Glamping Lakeside Rancabali
- Description: Glamping Lakeside Rancabali menawarkan tempat berkemah mewah yang cocok untuk Anda jadikan destinasi liburan di akhir pekan. Seperti namanya, glamping atau glamour camping terdiri dari tenda yang berisikan tempat tidur nyaman, kamar mandi di dalam ruangan, pemanas, dan berbagai amenitas layaknya hotel berbintang. Berlokasi di Jalan Raya Ciwidey Rancabali KM 1, Glamping Lakeside berada di sekitar Situ Patenggang dan perkebunan teh yang subur. Tidak heran jika pemandangan dari glamping ini begitu apik dan Instagramable untuk berfoto.
- City: Bandung
- Price: 30000
- Rating: 4.40

- Recommended Category: 92, Estimated Rating: 3.97
- Places in category '92' with similar rating ---
- - No similar places found


**Top Recommendations for User 5**
- Recommended Category: 90, Estimated Rating: 4.12
- Places in category '90' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 4.10
- Places in category '83' with similar rating ---
- Place: 83
- Place: Alive Museum Ancol
- Description: Museum kini tidak hanya menawarkan benda ‚Äì benda yang bernilai sejarah dan edukatif. Tapi telah disulap menjadi wahana rekreasi seru dengan objek foto yang berbeda. Seperti Alive Museum yang ada di Ancol. Alive Museum menampilkan berbagai seni rupa berupa lukisan 3 dimensi. Dimana pengunjung dapat berfoto dengan latar belakang yang nyata meski hanya dari foto. Dan karena 3 dimensi, tentu foto akan terlihat nyata dengan objeknya.
- City: Jakarta
- Price: 200000
- Rating: 4.30

- Recommended Category: 157, Estimated Rating: 4.09
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 416, Estimated Rating: 3.98
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.97
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20


**Top Recommendations for User 6**
- Recommended Category: 416, Estimated Rating: 4.52
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 97, Estimated Rating: 4.24
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 157, Estimated Rating: 4.21
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.18
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 112, Estimated Rating: 4.16
- Places in category '112' with similar rating ---
- Place: 112
- Place: Bukit Bintang Yogyakarta
- Description: Bukit Bintang merupakan salah satu lokasi nongkrong favorit di Yogyakarta. Saat malam tiba, pemandangan Yogyakarta sangatlah indah!Terletak di perbatasan Bantul dan Gunungkidul, siapa pun yang berkunjung ke kawasan ini dapat menikmati taburan gemintang di langit malam serta kerlip benderang lampu kota dari ketinggian.Kota ini memiliki Bukit Bintang yang selalu ramai dipadati kawula muda untuk menikmati keindahan malam. Bukit Bintang menjadi tempat yang sempurna untuk menikmati senja hingga malam tiba._x000D_
- City: Yogyakarta
- Price: 25000
- Rating: 4.50


**Top Recommendations for User 7**
- Recommended Category: 416, Estimated Rating: 4.57
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 387, Estimated Rating: 4.51
- Places in category '387' with similar rating ---
- Place: 387
- Place: Obyek Wisata Goa Kreo
- Description: Goa Kreo Semarang yang berada di ibukota Jawa Tengah ini begitu hit di kalangan netizen. Tak heran jika cuaca cerah, antrian menuju ke sana begitu mengular. Meski tak ada angkutan umum menuju ke sana, tak menghalangi rasa penasaran wisatawan. Gua ini terbentuk secara alami yang membedakan gua ini dengan gua lainnya adalah letaknya. lokasinya terletak di tengah waduk jatibarang, sebuah bendungan yang membendung Sungai Kreo.
- City: Semarang
- Price: 5500
- Rating: 4.30

- Recommended Category: 279, Estimated Rating: 4.48
- Places in category '279' with similar rating ---
- Place: 279
- Place: Masjid Agung Trans Studio Bandung
- Description: Masjid Agung Trans Studio Bandung (TSB) berdiri megah. Rumah ibadah seluas 4.000 meter persegi bergaya Timur Tengah ini menjadi oase di tengah-tengah pusat perbelanjaan dan tempat rekreasi. konsep masjid modern yang mengadopsi ala Timur Tengah ini berdasarkan arahan Chairul Tanjung (CT), selaku pemiki CT Corp. Dia langsung menerawang guna mewujudkan desain Masjid Agung TSB yang kental dengan sentuhan Masjid Nabawi.
- City: Bandung
- Price: 0
- Rating: 4.80

- Recommended Category: 232, Estimated Rating: 4.47
- Places in category '232' with similar rating ---
- Place: 232
- Place: Bukit Moko
- Description: Bandung sebagai destinasi wisata tak pernah ada habisnya. Didukung dengan lanskap yang cantik, kawasan Bandung mampu menarik perhatian wisatawan. Baik dari segi alam, budaya, kuliner, dan seni kreatif secara bersamaan. Dari sekian banyak tempat wisata yang tersedia, Bukit Moko Bandung menjadi salah satu yang cukup populer namanya dalam beberapa tahun belakangan. Berada di ketinggian sekitar 1500 mdpl, Bukit Moko memiliki cuaca yang sejuk. Bagi pengunjung yang tidak biasa di cuaca ini, ada baiknya membawa jaket tebal apalagi jika datang saat musim penghujan.
- City: Bandung
- Price: 25000
- Rating: 4.50

- Recommended Category: 322, Estimated Rating: 4.43
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20


**Top Recommendations for User 8**
- Recommended Category: 332, Estimated Rating: 4.25
- Places in category '332' with similar rating ---
- Place: 332
- Place: Rainbow Garden
- Description: Rainbow Garden Harapan Indah salah satu taman rekreasi yang terutama cocok untuk keluarga. Berkonsep ‚Äòeat and play‚Äô, Rainbow Garden Bekasi memadukan arena permainan anak dengan wisata kuliner. Berbagai wahana dan aktivitas tersedia di taman wisata penuh warna seluas 3000 m2 ini.
- City: Bandung
- Price: 20000
- Rating: 4.60

- Recommended Category: 431, Estimated Rating: 4.23
- Places in category '431' with similar rating ---
- Place: 431
- Place: Taman Hiburan Rakyat
- Description: Taman Hiburan Rakyat atau THR tentunya sudah tak asing lagi bagi masyarakat Surabaya. Taman ini berletak di belakang Taman Remaja Surabaya (TRS) dan juga ada di belakang Hi-Tech Mall. THR biasanya digunakan untuk pertunjukan kesenian daerah. Taman ini menjadi ikon Surabaya karena sejarahnya yang cukup menarik.
- City: Surabaya
- Price: 5000
- Rating: 4.20

- Recommended Category: 134, Estimated Rating: 4.19
- Places in category '134' with similar rating ---
- Place: 134
- Place: Desa Wisata Gamplong
- Description: Desa Wisata Gamplong adalah desa wisata kerajinan tenun yang berada di Padukuhan Gamplong Desa Sumber Rahayu Kecamatan Moyudan Kabupaten Sleman, Yogyakarta. Desa wisata yang terletak di sebelah barat Kota Yogyakarta ini cukup menarik untuk disinggahi wisatawan terlebih karena masih adanya industri kerajinan tenun tradisional dengan menggunakan Alat Tenun Bukan Mesin (ATBM). Dengan ATBM, masyarakat perajin Gamplong mampu menghasilkan kain tenun sebagai bahan stagen (kain panjang untuk melilit bagian perut wanita). Selain kerajinan tenun, perajin juga mampu memproduksi kerajinan anyaman untuk suvenir.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 416, Estimated Rating: 4.17
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 232, Estimated Rating: 4.15
- Places in category '232' with similar rating ---
- Place: 232
- Place: Bukit Moko
- Description: Bandung sebagai destinasi wisata tak pernah ada habisnya. Didukung dengan lanskap yang cantik, kawasan Bandung mampu menarik perhatian wisatawan. Baik dari segi alam, budaya, kuliner, dan seni kreatif secara bersamaan. Dari sekian banyak tempat wisata yang tersedia, Bukit Moko Bandung menjadi salah satu yang cukup populer namanya dalam beberapa tahun belakangan. Berada di ketinggian sekitar 1500 mdpl, Bukit Moko memiliki cuaca yang sejuk. Bagi pengunjung yang tidak biasa di cuaca ini, ada baiknya membawa jaket tebal apalagi jika datang saat musim penghujan.
- City: Bandung
- Price: 25000
- Rating: 4.50


**Top Recommendations for User 9**
- Recommended Category: 44, Estimated Rating: 4.54
- Places in category '44' with similar rating ---
- Place: 44
- Place: Monumen Selamat Datang
- Description: Monumen Selamat Datang adalah sebuah monumen yang terletak di tengah Bundaran Hotel Indonesia, Jakarta, Indonesia. Monumen ini berupa patung sepasang manusia yang sedang menggenggam bunga dan melambaikan tangan. Patung tersebut menghadap ke utara yang berarti mereka menyambut orang-orang yang datang dari arah Monumen Nasional._x000D_
- City: Jakarta
- Price: 0
- Rating: 4.70

- Recommended Category: 333, Estimated Rating: 4.40
- Places in category '333' with similar rating ---
- Place: 333
- Place: Kota Mini
- Description: Destinasi yang sangat menarik bernuansa eropa lengkap dengan fasilitas publiknya membuat anda seolah-olah sedang berada di eropa. Ada rumah sakit, kantor polisi, box telepon, cafe-caf√© pinggir jalan, dan bangunan-bangunan klasik lainnya. Masih satu kompleks dengan Floating Market, tempat wisata ini menyajikan arsitektur eropa klasik yang begitu kental, dijamin anda akan enggan beranjak.
- City: Bandung
- Price: 20000
- Rating: 4.40

- Recommended Category: 90, Estimated Rating: 4.36
- Places in category '90' with similar rating ---
- Place: 90
- Place: Kampung Wisata Taman Sari
- Description: Taman Sari Yogyakarta atau Taman Sari Keraton Yogyakarta (Hanacaraka:Í¶†Í¶©Í¶§ÍßÄÍ¶±Í¶´Í¶∂Í¶îÍ¶™Í¶∫Í¶¥Í¶íÍ¶æÍ¶èÍ¶ÇÍ¶°, Tamansari Ngayogyakarta) adalah situs bekas taman atau kebun istana Keraton Ngayogyakarta Hadiningrat, yang dapat dibandingkan dengan Kebun Raya Bogor sebagai kebun Istana Bogor. Kebun ini dibangun pada zaman Sultan Hamengku Buwono I (HB I) pada tahun 1758-1765/9. Awalnya, taman yang mendapat sebutan "The Fragrant Garden" ini memiliki luas lebih dari 10 hektare dengan sekitar 57 bangunan baik berupa gedung, kolam pemandian, jembatan gantung, kanal air, maupun danau buatan beserta pulau buatan dan lorong bawah air. Kebun yang digunakan secara efektif antara 1765-1812 ini pada mulanya membentang dari barat daya kompleks Kedhaton sampai tenggara kompleks Magangan. Namun saat ini, sisa-sisa bagian Taman Sari yang dapat dilihat hanyalah yang berada di barat daya kompleks Kedhaton saja.
- City: Yogyakarta
- Price: 5000
- Rating: 4.60

- Recommended Category: 401, Estimated Rating: 4.32
- Places in category '401' with similar rating ---
- Place: 401
- Place: Taman Keputran
- Description: Ntah, mengapa nama taman ini disebut dengan taman keputran, namun jika dikaitkan dengan lokasi taman, berada di persimpangan jalan Keputran Surabaya. Selain itu, di seberang sungai terdapat juga sebuah taman yang bernama Taman Lalu Lintas Surabaya. Keunikan Taman Keputran dibandingkan dengan taman lainnya adalah adanya lampu yang kerangka tiangnya mirip dengan manusia, dengan gaya pose yang berbeda-beda itu, membuat taman ini terlihat ada sentuhan kreatif dan inovatif. Lampu-lampu itu terang dan indah pada malam hari. Untuknya, taman ini cukup representatif sebagai pilihan mendapatkan tempat tongkrongan yang tenang pada malam hari. Tak hanya itu, jalur pijat refleksi kaki juga tersedia disana. Cocok untuk pengunjung yang ingin mencoba pengobatan refleksi kaki sederhana. Melalui pemahaman tentang jalur refleksi secara mendalam, sebagian orang memilih sarana ini dalam usaha menyembuhkan penyakit._x000D_
- City: Surabaya
- Price: 0
- Rating: 4.30

- Recommended Category: 164, Estimated Rating: 4.27
- Places in category '164' with similar rating ---
- Place: 164
- Place: Pintoe Langit Dahromo
- Description: Pintu Langit Dahromo ini menyediakan berbagai spot selfie yang hitz dan instagramable dengan latar belakang panorama keindahan sebagian Kota Jogja yang istimewa. Adapun berbagai spot foto tersebut seperti spot foto rumah dengan bunga-bunga disekitarnya, pintu langit, gardu pandang yang berbentuk love, dan ada juga sayap capung. Pintu Langit Dahromo berlokasi di Jl. Dahromo, Karang Asem, Muntuk, Dlingo, Bantul, Daerah Istimewa Yogyakarta 55783. Tepatnya berada di jalur wisata perbukitan sebelah selatan bukit Lintang Sewu. Untuk mencapai Pintu Langit ini, rute perjalanan yang dilalui cukup mudah karena berada dekat dengan wisata lainnya.
- City: Yogyakarta
- Price: 2500
- Rating: 4.40


**Top Recommendations for User 10**
- Recommended Category: 416, Estimated Rating: 4.33
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 139, Estimated Rating: 4.32
- Places in category '139' with similar rating ---
- Place: 139
- Place: Puncak Gunung Api Purba - Nglanggeran
- Description: Gunung Nglanggeran adalah sebuah gunung di Daerah Istimewa Yogyakarta, Indonesia. Gunung ini merupakan suatu gunung api purba yang terbentuk sekitar 0,6-70 juta tahun yang lalu atau yang memiliki umur tersier (Oligo-Miosen). Gunung Nglanggeran memiliki batuan yang sangat khas karena didominasi oleh aglomerat dan breksi gunung api. Gunung ini terletak di Desa Nglanggeran, Kecamatan Patuk, Kabupaten Gunung Kidul yang berada pada deretan Pegunungan Baturagung.
- City: Yogyakarta
- Price: 10000
- Rating: 4.70

- Recommended Category: 157, Estimated Rating: 4.31
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 200, Estimated Rating: 4.27
- Places in category '200' with similar rating ---
- Place: 200
- Place: Pantai Watu Kodok
- Description: Pantai Watu Kodok merupakan salah satu pantai yang ada di Gunung Kidul. Pantai Watu Kodok ini memiliki pasir putih dan air laut yang masih berwarna biru yang sangat indah. Di pantai ini terdapat beberapa batu karang yang terjal, batu tersebut menjadi daya tarik para wisatawan dan keunikan bagi pantai itu sendiri. Dan suasana pantai yang masih alami juga menjadikan pantai ini menjadi daya tariknya, tetapi pantai ini masih tergolong pantai baru jadi masih belum banyak wisatawan. Pantai Watu Kodok ini sangat cocok bagi Anda yang ingin menghabiskan waktu liburan bersama keluarga.
- City: Yogyakarta
- Price: 5000
- Rating: 4.60

- Recommended Category: 254, Estimated Rating: 4.26
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30


**Top Recommendations for User 11**
- Recommended Category: 138, Estimated Rating: 3.86
- Places in category '138' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.85
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.79
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.74
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.71
- Places in category '52' with similar rating ---
- - No similar places found


**Top Recommendations for User 12**
- Recommended Category: 416, Estimated Rating: 4.72
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 279, Estimated Rating: 4.57
- Places in category '279' with similar rating ---
- Place: 279
- Place: Masjid Agung Trans Studio Bandung
- Description: Masjid Agung Trans Studio Bandung (TSB) berdiri megah. Rumah ibadah seluas 4.000 meter persegi bergaya Timur Tengah ini menjadi oase di tengah-tengah pusat perbelanjaan dan tempat rekreasi. konsep masjid modern yang mengadopsi ala Timur Tengah ini berdasarkan arahan Chairul Tanjung (CT), selaku pemiki CT Corp. Dia langsung menerawang guna mewujudkan desain Masjid Agung TSB yang kental dengan sentuhan Masjid Nabawi.
- City: Bandung
- Price: 0
- Rating: 4.80

- Recommended Category: 90, Estimated Rating: 4.55
- Places in category '90' with similar rating ---
- Place: 90
- Place: Kampung Wisata Taman Sari
- Description: Taman Sari Yogyakarta atau Taman Sari Keraton Yogyakarta (Hanacaraka:Í¶†Í¶©Í¶§ÍßÄÍ¶±Í¶´Í¶∂Í¶îÍ¶™Í¶∫Í¶¥Í¶íÍ¶æÍ¶èÍ¶ÇÍ¶°, Tamansari Ngayogyakarta) adalah situs bekas taman atau kebun istana Keraton Ngayogyakarta Hadiningrat, yang dapat dibandingkan dengan Kebun Raya Bogor sebagai kebun Istana Bogor. Kebun ini dibangun pada zaman Sultan Hamengku Buwono I (HB I) pada tahun 1758-1765/9. Awalnya, taman yang mendapat sebutan "The Fragrant Garden" ini memiliki luas lebih dari 10 hektare dengan sekitar 57 bangunan baik berupa gedung, kolam pemandian, jembatan gantung, kanal air, maupun danau buatan beserta pulau buatan dan lorong bawah air. Kebun yang digunakan secara efektif antara 1765-1812 ini pada mulanya membentang dari barat daya kompleks Kedhaton sampai tenggara kompleks Magangan. Namun saat ini, sisa-sisa bagian Taman Sari yang dapat dilihat hanyalah yang berada di barat daya kompleks Kedhaton saja.
- City: Yogyakarta
- Price: 5000
- Rating: 4.60

- Recommended Category: 254, Estimated Rating: 4.55
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 263, Estimated Rating: 4.49
- Places in category '263' with similar rating ---
- Place: 263
- Place: Curug Batu Templek
- Description: Curug Batu Templek Bandung adalah sebuah wisata alam air terjun yang terletak di Kota Bandung Timur. Dari sekian banyak wisata alam air terjun di Bandung, Curug Batu Templek tak kalah menarik karena pemandangan di sekitarnya dipenuhi dengan terbing yang berbatu cadas. Sejarah Curug Batu Templek sangat sederhana, hanya karena di tempat ini dahulunya terdapat penambangan batu yang terdapat sebuah aliran air terjun dan diketahui aliran tersebut berasal dari sungai yang ada di atas tebing sehingga dinamakan Curug Batu Templek.
- City: Bandung
- Price: 5000
- Rating: 4.10


**Top Recommendations for User 13**
- Recommended Category: 416, Estimated Rating: 4.11
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 210, Estimated Rating: 4.02
- Places in category '210' with similar rating ---
- Place: 210
- Place: Pantai Congot
- Description: Selain Pantai Glagah dan Pantai Trisik, ternyata masih ada satu lagi pantai yang berada di Kulon Progo. Yakni Pantai Congot, pantai yang satu ini terkenal sebagai surganya memancing ikan. Maka tak heran jika Pantai Congot banyak dikunjungi oleh mereka para mancing mania, selain itu masyarakat sekitar Pantai Congot menggantungkan hidupnya sebagai nelayan. Pantai Congot memang tak sepopuler pantai lain yang ada di Kulon Progo, mulai banyak dikunjungi wisatawan ketika dibukanya Hutan Mangrove Pasir Mendit. Karena memang lokasinya yang tak terlalu jauh dari Pantai Congot, yakni berada di sebelah barat pantai. Pantai Congot merupakan muara dari Sungai Bogowonto. Ini pulalah yang menyebabkan Pantai Congot menjadi spot favorite para pemancing, karena terdapat banyak jenis ikan yang ada mulai dari ikan air tawar, ikan air payau hingga ikan air asin.
- City: Yogyakarta
- Price: 3000
- Rating: 4.30

- Recommended Category: 279, Estimated Rating: 4.00
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 200, Estimated Rating: 3.98
- Places in category '200' with similar rating ---
- - No similar places found

- Recommended Category: 160, Estimated Rating: 3.97
- Places in category '160' with similar rating ---
- - No similar places found


**Top Recommendations for User 14**
- Recommended Category: 97, Estimated Rating: 4.25
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 139, Estimated Rating: 4.23
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 136, Estimated Rating: 4.14
- Places in category '136' with similar rating ---
- Place: 136
- Place: Grojogan Watu Purbo Bangunrejo
- Description: Objek wisata itu tak adalah Grojogan Watu Purbo yang berada di Bangunrejo, Merdikorejo, Kecamatan Tempel. Objek wisata itu sekitar setahun terakhir cukup populer di kalangan wisatawan karena memiliki pemandangan eksotis berupa air terjun yang memiliki enam tingkatan. Wisatawan yang datang rata-rata menjadikan air terjun itu sebagai latar untuk swafoto karena pemandangannya yang dinilai instagramable. Grojokan Watu Purbo ini tepatnya berlokasi di aliran Kali Krisak, yang merupakan jalur dari lahar dingin yang mengalir dari Gunung Merapi. Pemandangan kawasan ini eksotis karena dikepung pepohonan asri serta hamparan sawah. Munculnya air terjun atau grojogan ini berasal dari enam dam dengan ketinggian bervariasi tak lebih dari 10 meter.
- City: Yogyakarta
- Price: 10000
- Rating: 4.50

- Recommended Category: 314, Estimated Rating: 4.13
- Places in category '314' with similar rating ---
- Place: 314
- Place: Tafso Barn
- Description: Nama Punclut mungkin sudah cukup akrab di telinga wisatawan. Kawasan dataran tinggi di Bandung yang belakangan populer sebagai destinasi wisata. Selain menawarkan pemandangan yang indah, kawasan ini juga memiliki banyak tempat wisata untuk rekreasi. Salah satunya adalah kawasan wisata sekaligus restoran Tafso dan Boda Barn atau sering dikenal juga dengan Tafso Barn. Tempat rekreasi ini memadukan tempat makan dengan nuansa taman dan gudang yang unik. Pemandangan perbukitan yang dipotong garis horison menjadi incaran utama pengunjung. Sangat tepat nongkrong dan bersantai ketika berlibur di Bandung.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 279, Estimated Rating: 4.13
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 15**
- Recommended Category: 254, Estimated Rating: 4.31
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 322, Estimated Rating: 4.25
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 416, Estimated Rating: 4.13
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 112, Estimated Rating: 4.10
- Places in category '112' with similar rating ---
- Place: 112
- Place: Bukit Bintang Yogyakarta
- Description: Bukit Bintang merupakan salah satu lokasi nongkrong favorit di Yogyakarta. Saat malam tiba, pemandangan Yogyakarta sangatlah indah!Terletak di perbatasan Bantul dan Gunungkidul, siapa pun yang berkunjung ke kawasan ini dapat menikmati taburan gemintang di langit malam serta kerlip benderang lampu kota dari ketinggian.Kota ini memiliki Bukit Bintang yang selalu ramai dipadati kawula muda untuk menikmati keindahan malam. Bukit Bintang menjadi tempat yang sempurna untuk menikmati senja hingga malam tiba._x000D_
- City: Yogyakarta
- Price: 25000
- Rating: 4.50

- Recommended Category: 253, Estimated Rating: 4.05
- Places in category '253' with similar rating ---
- - No similar places found


**Top Recommendations for User 16**
- Recommended Category: 416, Estimated Rating: 4.12
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 251, Estimated Rating: 4.11
- Places in category '251' with similar rating ---
- Place: 251
- Place: Taman Lansia
- Description: Berlibur santai di akhir pekan cocok dilakukan dengan menghabiskan waktu di taman. Salah satu taman yang dapat menjadi tujuan wisata adalah Taman Lansia Bandung. Lansia merupakan singkatan dari Lanjut Usia. Meski begitu, taman ini tidak dikhususkan untuk para lansia, namun untuk semua kalangan. Nama lansia kemungkinan diberikan karena usia taman ini yang sudah sangat tua. Bahkan, usianya sudah ratusan tahun, karena sudah ada sejak tahun 1885.
- City: Bandung
- Price: 0
- Rating: 4.40

- Recommended Category: 97, Estimated Rating: 4.11
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 79, Estimated Rating: 4.09
- Places in category '79' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 4.06
- Places in category '1' with similar rating ---
- - No similar places found


**Top Recommendations for User 17**
- Recommended Category: 112, Estimated Rating: 4.25
- Places in category '112' with similar rating ---
- Place: 112
- Place: Bukit Bintang Yogyakarta
- Description: Bukit Bintang merupakan salah satu lokasi nongkrong favorit di Yogyakarta. Saat malam tiba, pemandangan Yogyakarta sangatlah indah!Terletak di perbatasan Bantul dan Gunungkidul, siapa pun yang berkunjung ke kawasan ini dapat menikmati taburan gemintang di langit malam serta kerlip benderang lampu kota dari ketinggian.Kota ini memiliki Bukit Bintang yang selalu ramai dipadati kawula muda untuk menikmati keindahan malam. Bukit Bintang menjadi tempat yang sempurna untuk menikmati senja hingga malam tiba._x000D_
- City: Yogyakarta
- Price: 25000
- Rating: 4.50

- Recommended Category: 232, Estimated Rating: 4.21
- Places in category '232' with similar rating ---
- Place: 232
- Place: Bukit Moko
- Description: Bandung sebagai destinasi wisata tak pernah ada habisnya. Didukung dengan lanskap yang cantik, kawasan Bandung mampu menarik perhatian wisatawan. Baik dari segi alam, budaya, kuliner, dan seni kreatif secara bersamaan. Dari sekian banyak tempat wisata yang tersedia, Bukit Moko Bandung menjadi salah satu yang cukup populer namanya dalam beberapa tahun belakangan. Berada di ketinggian sekitar 1500 mdpl, Bukit Moko memiliki cuaca yang sejuk. Bagi pengunjung yang tidak biasa di cuaca ini, ada baiknya membawa jaket tebal apalagi jika datang saat musim penghujan.
- City: Bandung
- Price: 25000
- Rating: 4.50

- Recommended Category: 322, Estimated Rating: 4.20
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 138, Estimated Rating: 4.20
- Places in category '138' with similar rating ---
- Place: 138
- Place: Jogja Exotarium
- Description: Di Yogyakarta, tepatnya di Sleman, ada satu tempat wisata edukasi yang patut dikunjungi. Namanya adalah Jogja Exotarium ‚Äî terdengar unik, kan? Namun sebenarnya, tempat ini merupakan taman hewan berskala kecil. Koleksi hewannya beragam dan bisa diajak berinteraksi secara langsung
- City: Yogyakarta
- Price: 20000
- Rating: 4.40

- Recommended Category: 254, Estimated Rating: 4.19
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30


**Top Recommendations for User 18**
- Recommended Category: 279, Estimated Rating: 4.27
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 4.26
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 136, Estimated Rating: 4.23
- Places in category '136' with similar rating ---
- Place: 136
- Place: Grojogan Watu Purbo Bangunrejo
- Description: Objek wisata itu tak adalah Grojogan Watu Purbo yang berada di Bangunrejo, Merdikorejo, Kecamatan Tempel. Objek wisata itu sekitar setahun terakhir cukup populer di kalangan wisatawan karena memiliki pemandangan eksotis berupa air terjun yang memiliki enam tingkatan. Wisatawan yang datang rata-rata menjadikan air terjun itu sebagai latar untuk swafoto karena pemandangannya yang dinilai instagramable. Grojokan Watu Purbo ini tepatnya berlokasi di aliran Kali Krisak, yang merupakan jalur dari lahar dingin yang mengalir dari Gunung Merapi. Pemandangan kawasan ini eksotis karena dikepung pepohonan asri serta hamparan sawah. Munculnya air terjun atau grojogan ini berasal dari enam dam dengan ketinggian bervariasi tak lebih dari 10 meter.
- City: Yogyakarta
- Price: 10000
- Rating: 4.50

- Recommended Category: 416, Estimated Rating: 4.22
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 387, Estimated Rating: 4.21
- Places in category '387' with similar rating ---
- Place: 387
- Place: Obyek Wisata Goa Kreo
- Description: Goa Kreo Semarang yang berada di ibukota Jawa Tengah ini begitu hit di kalangan netizen. Tak heran jika cuaca cerah, antrian menuju ke sana begitu mengular. Meski tak ada angkutan umum menuju ke sana, tak menghalangi rasa penasaran wisatawan. Gua ini terbentuk secara alami yang membedakan gua ini dengan gua lainnya adalah letaknya. lokasinya terletak di tengah waduk jatibarang, sebuah bendungan yang membendung Sungai Kreo.
- City: Semarang
- Price: 5500
- Rating: 4.30


**Top Recommendations for User 19**
- Recommended Category: 321, Estimated Rating: 4.49
- Places in category '321' with similar rating ---
- Place: 321
- Place: Glamping Lakeside Rancabali
- Description: Glamping Lakeside Rancabali menawarkan tempat berkemah mewah yang cocok untuk Anda jadikan destinasi liburan di akhir pekan. Seperti namanya, glamping atau glamour camping terdiri dari tenda yang berisikan tempat tidur nyaman, kamar mandi di dalam ruangan, pemanas, dan berbagai amenitas layaknya hotel berbintang. Berlokasi di Jalan Raya Ciwidey Rancabali KM 1, Glamping Lakeside berada di sekitar Situ Patenggang dan perkebunan teh yang subur. Tidak heran jika pemandangan dari glamping ini begitu apik dan Instagramable untuk berfoto.
- City: Bandung
- Price: 30000
- Rating: 4.40

- Recommended Category: 279, Estimated Rating: 4.48
- Places in category '279' with similar rating ---
- Place: 279
- Place: Masjid Agung Trans Studio Bandung
- Description: Masjid Agung Trans Studio Bandung (TSB) berdiri megah. Rumah ibadah seluas 4.000 meter persegi bergaya Timur Tengah ini menjadi oase di tengah-tengah pusat perbelanjaan dan tempat rekreasi. konsep masjid modern yang mengadopsi ala Timur Tengah ini berdasarkan arahan Chairul Tanjung (CT), selaku pemiki CT Corp. Dia langsung menerawang guna mewujudkan desain Masjid Agung TSB yang kental dengan sentuhan Masjid Nabawi.
- City: Bandung
- Price: 0
- Rating: 4.80

- Recommended Category: 416, Estimated Rating: 4.46
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 399, Estimated Rating: 4.29
- Places in category '399' with similar rating ---
- Place: 399
- Place: Taman Pelangi
- Description: Kalau pelangi biasanya ada di siang hari pasca hujan, maka di Taman Pelangi Yogyakarta pengunjung justru bisa menikmatinya setiap malam hari. Ya, ini lantaran taman ini merupakan Taman Lampion beraneka warna dan rupa. Buka sejak petang, Taman ini memang sangat cocok dijadikan pilihan destinasi rekreasi keluarga saat malam hari. Namun karena merupakan taman outdoor alias terbuka, maka pilihan paling tepat untuk datang ialah saat cuaca tidak hujan.
- City: Surabaya
- Price: 0
- Rating: 4.50

- Recommended Category: 1, Estimated Rating: 4.28
- Places in category '1' with similar rating ---
- Place: 1
- Place: Monumen Nasional
- Description: Monumen Nasional atau yang populer disingkat dengan Monas atau Tugu Monas adalah monumen peringatan setinggi 132 meter (433 kaki) yang didirikan untuk mengenang perlawanan dan perjuangan rakyat Indonesia untuk merebut kemerdekaan dari pemerintahan kolonial Hindia Belanda. Pembangunan monumen ini dimulai pada tanggal 17 Agustus 1961 di bawah perintah presiden Soekarno dan dibuka untuk umum pada tanggal 12 Juli 1975. Tugu ini dimahkotai lidah api yang dilapisi lembaran emas yang melambangkan semangat perjuangan yang menyala-nyala. Monumen Nasional terletak tepat di tengah Lapangan Medan Merdeka, Jakarta Pusat.
- City: Jakarta
- Price: 20000
- Rating: 4.60


**Top Recommendations for User 20**
- Recommended Category: 139, Estimated Rating: 3.94
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.90
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.86
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 91, Estimated Rating: 3.83
- Places in category '91' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.82
- Places in category '1' with similar rating ---
- - No similar places found


**Top Recommendations for User 21**
- Recommended Category: 422, Estimated Rating: 4.10
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 167, Estimated Rating: 4.04
- Places in category '167' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 4.02
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 4.02
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 321, Estimated Rating: 4.00
- Places in category '321' with similar rating ---
- Place: 321
- Place: Glamping Lakeside Rancabali
- Description: Glamping Lakeside Rancabali menawarkan tempat berkemah mewah yang cocok untuk Anda jadikan destinasi liburan di akhir pekan. Seperti namanya, glamping atau glamour camping terdiri dari tenda yang berisikan tempat tidur nyaman, kamar mandi di dalam ruangan, pemanas, dan berbagai amenitas layaknya hotel berbintang. Berlokasi di Jalan Raya Ciwidey Rancabali KM 1, Glamping Lakeside berada di sekitar Situ Patenggang dan perkebunan teh yang subur. Tidak heran jika pemandangan dari glamping ini begitu apik dan Instagramable untuk berfoto.
- City: Bandung
- Price: 30000
- Rating: 4.40


**Top Recommendations for User 22**
- Recommended Category: 416, Estimated Rating: 4.02
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 112, Estimated Rating: 3.97
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 146, Estimated Rating: 3.96
- Places in category '146' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.88
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.88
- Places in category '314' with similar rating ---
- Place: 314
- Place: Tafso Barn
- Description: Nama Punclut mungkin sudah cukup akrab di telinga wisatawan. Kawasan dataran tinggi di Bandung yang belakangan populer sebagai destinasi wisata. Selain menawarkan pemandangan yang indah, kawasan ini juga memiliki banyak tempat wisata untuk rekreasi. Salah satunya adalah kawasan wisata sekaligus restoran Tafso dan Boda Barn atau sering dikenal juga dengan Tafso Barn. Tempat rekreasi ini memadukan tempat makan dengan nuansa taman dan gudang yang unik. Pemandangan perbukitan yang dipotong garis horison menjadi incaran utama pengunjung. Sangat tepat nongkrong dan bersantai ketika berlibur di Bandung.
- City: Bandung
- Price: 0
- Rating: 4.20


**Top Recommendations for User 23**
- Recommended Category: 416, Estimated Rating: 4.28
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.24
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 321, Estimated Rating: 4.13
- Places in category '321' with similar rating ---
- Place: 321
- Place: Glamping Lakeside Rancabali
- Description: Glamping Lakeside Rancabali menawarkan tempat berkemah mewah yang cocok untuk Anda jadikan destinasi liburan di akhir pekan. Seperti namanya, glamping atau glamour camping terdiri dari tenda yang berisikan tempat tidur nyaman, kamar mandi di dalam ruangan, pemanas, dan berbagai amenitas layaknya hotel berbintang. Berlokasi di Jalan Raya Ciwidey Rancabali KM 1, Glamping Lakeside berada di sekitar Situ Patenggang dan perkebunan teh yang subur. Tidak heran jika pemandangan dari glamping ini begitu apik dan Instagramable untuk berfoto.
- City: Bandung
- Price: 30000
- Rating: 4.40

- Recommended Category: 52, Estimated Rating: 4.12
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 279, Estimated Rating: 4.10
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 24**
- Recommended Category: 431, Estimated Rating: 4.02
- Places in category '431' with similar rating ---
- Place: 431
- Place: Taman Hiburan Rakyat
- Description: Taman Hiburan Rakyat atau THR tentunya sudah tak asing lagi bagi masyarakat Surabaya. Taman ini berletak di belakang Taman Remaja Surabaya (TRS) dan juga ada di belakang Hi-Tech Mall. THR biasanya digunakan untuk pertunjukan kesenian daerah. Taman ini menjadi ikon Surabaya karena sejarahnya yang cukup menarik.
- City: Surabaya
- Price: 5000
- Rating: 4.20

- Recommended Category: 134, Estimated Rating: 4.00
- Places in category '134' with similar rating ---
- Place: 134
- Place: Desa Wisata Gamplong
- Description: Desa Wisata Gamplong adalah desa wisata kerajinan tenun yang berada di Padukuhan Gamplong Desa Sumber Rahayu Kecamatan Moyudan Kabupaten Sleman, Yogyakarta. Desa wisata yang terletak di sebelah barat Kota Yogyakarta ini cukup menarik untuk disinggahi wisatawan terlebih karena masih adanya industri kerajinan tenun tradisional dengan menggunakan Alat Tenun Bukan Mesin (ATBM). Dengan ATBM, masyarakat perajin Gamplong mampu menghasilkan kain tenun sebagai bahan stagen (kain panjang untuk melilit bagian perut wanita). Selain kerajinan tenun, perajin juga mampu memproduksi kerajinan anyaman untuk suvenir.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 416, Estimated Rating: 3.97
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.94
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 3.93
- Places in category '401' with similar rating ---
- Place: 401
- Place: Taman Keputran
- Description: Ntah, mengapa nama taman ini disebut dengan taman keputran, namun jika dikaitkan dengan lokasi taman, berada di persimpangan jalan Keputran Surabaya. Selain itu, di seberang sungai terdapat juga sebuah taman yang bernama Taman Lalu Lintas Surabaya. Keunikan Taman Keputran dibandingkan dengan taman lainnya adalah adanya lampu yang kerangka tiangnya mirip dengan manusia, dengan gaya pose yang berbeda-beda itu, membuat taman ini terlihat ada sentuhan kreatif dan inovatif. Lampu-lampu itu terang dan indah pada malam hari. Untuknya, taman ini cukup representatif sebagai pilihan mendapatkan tempat tongkrongan yang tenang pada malam hari. Tak hanya itu, jalur pijat refleksi kaki juga tersedia disana. Cocok untuk pengunjung yang ingin mencoba pengobatan refleksi kaki sederhana. Melalui pemahaman tentang jalur refleksi secara mendalam, sebagian orang memilih sarana ini dalam usaha menyembuhkan penyakit._x000D_
- City: Surabaya
- Price: 0
- Rating: 4.30


**Top Recommendations for User 25**
- Recommended Category: 279, Estimated Rating: 4.23
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 4.14
- Places in category '112' with similar rating ---
- Place: 112
- Place: Bukit Bintang Yogyakarta
- Description: Bukit Bintang merupakan salah satu lokasi nongkrong favorit di Yogyakarta. Saat malam tiba, pemandangan Yogyakarta sangatlah indah!Terletak di perbatasan Bantul dan Gunungkidul, siapa pun yang berkunjung ke kawasan ini dapat menikmati taburan gemintang di langit malam serta kerlip benderang lampu kota dari ketinggian.Kota ini memiliki Bukit Bintang yang selalu ramai dipadati kawula muda untuk menikmati keindahan malam. Bukit Bintang menjadi tempat yang sempurna untuk menikmati senja hingga malam tiba._x000D_
- City: Yogyakarta
- Price: 25000
- Rating: 4.50

- Recommended Category: 1, Estimated Rating: 4.13
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 4.12
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 232, Estimated Rating: 4.09
- Places in category '232' with similar rating ---
- - No similar places found


**Top Recommendations for User 26**
- Recommended Category: 295, Estimated Rating: 4.55
- Places in category '295' with similar rating ---
- Place: 295
- Place: Museum Nike Ardilla
- Description: Museum Nike Ardilla diresmikan atau dibuka untuk umum tepat di hari dimana almarhumah Nike Ardilla jika masih hidup berusia 21 tahun. Diresmikan langsung sendiri oleh pihak keluarga yaitu ayahnya sendiri (Almarhum) R.Eddy Kusnaedi dengan pemotongan kue ulang tahun, dan disaksikan langsung di depan komunitas Nike Ardilla Fans Club. Museum ini menempati sebuah bangunan Rumah yang akan menjadi saksi bisu serta menyimpan segala kenangan Nike Ardilla selama hidupnya, dan populer juga dengan sebutan Rumah Nike Ardilla (RNA). Sahabat traveler‚Äôs, Konsep dasar bangunan Rumah Nike Ardilla (RNA) ini adalah merupakan paduan desain yang menyerupai Planet Hollywood, Hard Rock Cafe dan Museum pada umumnya yang biasa kita lihat. Museum Nike Ardilla ini menempati bangunan 2 lantai seluas 500 meter persegi, dimana lantai pertama digunakan sebagai tempat tinggal keluarga almarhumah Nike Ardila. Sementara di lantai 2 Museum, selain digunakan sebagai ruang pamer barang koleksi Nike Ardilla, Aksesoris serta benda-benda pribadi miliknya, anda juga bisa menyaksikan seperti apa sih ruang pribadi berupa kamar Nike Ardilla dengan segala isinya.
- City: Bandung
- Price: 0
- Rating: 4.60

- Recommended Category: 157, Estimated Rating: 4.55
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.52
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 416, Estimated Rating: 4.49
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 279, Estimated Rating: 4.48
- Places in category '279' with similar rating ---
- Place: 279
- Place: Masjid Agung Trans Studio Bandung
- Description: Masjid Agung Trans Studio Bandung (TSB) berdiri megah. Rumah ibadah seluas 4.000 meter persegi bergaya Timur Tengah ini menjadi oase di tengah-tengah pusat perbelanjaan dan tempat rekreasi. konsep masjid modern yang mengadopsi ala Timur Tengah ini berdasarkan arahan Chairul Tanjung (CT), selaku pemiki CT Corp. Dia langsung menerawang guna mewujudkan desain Masjid Agung TSB yang kental dengan sentuhan Masjid Nabawi.
- City: Bandung
- Price: 0
- Rating: 4.80


**Top Recommendations for User 27**
- Recommended Category: 139, Estimated Rating: 4.27
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 4.17
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 97, Estimated Rating: 4.16
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 52, Estimated Rating: 4.12
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 322, Estimated Rating: 4.11
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20


**Top Recommendations for User 28**
- Recommended Category: 322, Estimated Rating: 4.46
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 98, Estimated Rating: 4.45
- Places in category '98' with similar rating ---
- Place: 98
- Place: Taman Pelangi Yogyakarta
- Description: Taman Pelangi Yogyakarta merupakan tempat wisata malam yang menampilkan warna-warni lampu lampion, sehingga terlihat seperti pelangi. Taman wisata ini terletak di Jalan Padjajaran (sebelumnya bernama Jalan Ring Road Utara), dan berada di lokasi Museum Monumen Yogya Kembali (Monumen Jogja Kembali) Yogyakarta. Taman Pelangi memiliki lebih dari 20 jenis permainan, 25 stand makan dan stand minum. Taman pelangi ini bisa dinikmati dari sore sampai malam, atau dari jam 17.00 sampai jam 23.00. Malam hari anda akan terasa lengkap dan sempurna di Taman Pelangi dengan banyaknya lampion yang menyala memberikan kesan yang menarik untuk menenangkan pikiran.
- City: Yogyakarta
- Price: 15000
- Rating: 4.30

- Recommended Category: 97, Estimated Rating: 4.43
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 332, Estimated Rating: 4.42
- Places in category '332' with similar rating ---
- Place: 332
- Place: Rainbow Garden
- Description: Rainbow Garden Harapan Indah salah satu taman rekreasi yang terutama cocok untuk keluarga. Berkonsep ‚Äòeat and play‚Äô, Rainbow Garden Bekasi memadukan arena permainan anak dengan wisata kuliner. Berbagai wahana dan aktivitas tersedia di taman wisata penuh warna seluas 3000 m2 ini.
- City: Bandung
- Price: 20000
- Rating: 4.60

- Recommended Category: 1, Estimated Rating: 4.39
- Places in category '1' with similar rating ---
- Place: 1
- Place: Monumen Nasional
- Description: Monumen Nasional atau yang populer disingkat dengan Monas atau Tugu Monas adalah monumen peringatan setinggi 132 meter (433 kaki) yang didirikan untuk mengenang perlawanan dan perjuangan rakyat Indonesia untuk merebut kemerdekaan dari pemerintahan kolonial Hindia Belanda. Pembangunan monumen ini dimulai pada tanggal 17 Agustus 1961 di bawah perintah presiden Soekarno dan dibuka untuk umum pada tanggal 12 Juli 1975. Tugu ini dimahkotai lidah api yang dilapisi lembaran emas yang melambangkan semangat perjuangan yang menyala-nyala. Monumen Nasional terletak tepat di tengah Lapangan Medan Merdeka, Jakarta Pusat.
- City: Jakarta
- Price: 20000
- Rating: 4.60


**Top Recommendations for User 29**
- Recommended Category: 254, Estimated Rating: 4.37
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 416, Estimated Rating: 4.21
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 112, Estimated Rating: 4.03
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 4.00
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 335, Estimated Rating: 3.95
- Places in category '335' with similar rating ---
- - No similar places found


**Top Recommendations for User 30**
- Recommended Category: 416, Estimated Rating: 4.15
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.06
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 134, Estimated Rating: 3.98
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.94
- Places in category '132' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.91
- Places in category '314' with similar rating ---
- Place: 314
- Place: Tafso Barn
- Description: Nama Punclut mungkin sudah cukup akrab di telinga wisatawan. Kawasan dataran tinggi di Bandung yang belakangan populer sebagai destinasi wisata. Selain menawarkan pemandangan yang indah, kawasan ini juga memiliki banyak tempat wisata untuk rekreasi. Salah satunya adalah kawasan wisata sekaligus restoran Tafso dan Boda Barn atau sering dikenal juga dengan Tafso Barn. Tempat rekreasi ini memadukan tempat makan dengan nuansa taman dan gudang yang unik. Pemandangan perbukitan yang dipotong garis horison menjadi incaran utama pengunjung. Sangat tepat nongkrong dan bersantai ketika berlibur di Bandung.
- City: Bandung
- Price: 0
- Rating: 4.20


**Top Recommendations for User 31**
- Recommended Category: 254, Estimated Rating: 4.12
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 164, Estimated Rating: 3.91
- Places in category '164' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.86
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.85
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 18, Estimated Rating: 3.85
- Places in category '18' with similar rating ---
- - No similar places found


**Top Recommendations for User 32**
- Recommended Category: 134, Estimated Rating: 4.04
- Places in category '134' with similar rating ---
- Place: 134
- Place: Desa Wisata Gamplong
- Description: Desa Wisata Gamplong adalah desa wisata kerajinan tenun yang berada di Padukuhan Gamplong Desa Sumber Rahayu Kecamatan Moyudan Kabupaten Sleman, Yogyakarta. Desa wisata yang terletak di sebelah barat Kota Yogyakarta ini cukup menarik untuk disinggahi wisatawan terlebih karena masih adanya industri kerajinan tenun tradisional dengan menggunakan Alat Tenun Bukan Mesin (ATBM). Dengan ATBM, masyarakat perajin Gamplong mampu menghasilkan kain tenun sebagai bahan stagen (kain panjang untuk melilit bagian perut wanita). Selain kerajinan tenun, perajin juga mampu memproduksi kerajinan anyaman untuk suvenir.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 1, Estimated Rating: 4.04
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.99
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.98
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.97
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30


**Top Recommendations for User 33**
- Recommended Category: 139, Estimated Rating: 4.09
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 4.06
- Places in category '321' with similar rating ---
- Place: 321
- Place: Glamping Lakeside Rancabali
- Description: Glamping Lakeside Rancabali menawarkan tempat berkemah mewah yang cocok untuk Anda jadikan destinasi liburan di akhir pekan. Seperti namanya, glamping atau glamour camping terdiri dari tenda yang berisikan tempat tidur nyaman, kamar mandi di dalam ruangan, pemanas, dan berbagai amenitas layaknya hotel berbintang. Berlokasi di Jalan Raya Ciwidey Rancabali KM 1, Glamping Lakeside berada di sekitar Situ Patenggang dan perkebunan teh yang subur. Tidak heran jika pemandangan dari glamping ini begitu apik dan Instagramable untuk berfoto.
- City: Bandung
- Price: 30000
- Rating: 4.40

- Recommended Category: 407, Estimated Rating: 3.98
- Places in category '407' with similar rating ---
- - No similar places found

- Recommended Category: 336, Estimated Rating: 3.93
- Places in category '336' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.92
- Places in category '300' with similar rating ---
- - No similar places found


**Top Recommendations for User 34**
- Recommended Category: 139, Estimated Rating: 4.38
- Places in category '139' with similar rating ---
- Place: 139
- Place: Puncak Gunung Api Purba - Nglanggeran
- Description: Gunung Nglanggeran adalah sebuah gunung di Daerah Istimewa Yogyakarta, Indonesia. Gunung ini merupakan suatu gunung api purba yang terbentuk sekitar 0,6-70 juta tahun yang lalu atau yang memiliki umur tersier (Oligo-Miosen). Gunung Nglanggeran memiliki batuan yang sangat khas karena didominasi oleh aglomerat dan breksi gunung api. Gunung ini terletak di Desa Nglanggeran, Kecamatan Patuk, Kabupaten Gunung Kidul yang berada pada deretan Pegunungan Baturagung.
- City: Yogyakarta
- Price: 10000
- Rating: 4.70

- Recommended Category: 157, Estimated Rating: 4.21
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 52, Estimated Rating: 4.21
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 134, Estimated Rating: 4.15
- Places in category '134' with similar rating ---
- Place: 134
- Place: Desa Wisata Gamplong
- Description: Desa Wisata Gamplong adalah desa wisata kerajinan tenun yang berada di Padukuhan Gamplong Desa Sumber Rahayu Kecamatan Moyudan Kabupaten Sleman, Yogyakarta. Desa wisata yang terletak di sebelah barat Kota Yogyakarta ini cukup menarik untuk disinggahi wisatawan terlebih karena masih adanya industri kerajinan tenun tradisional dengan menggunakan Alat Tenun Bukan Mesin (ATBM). Dengan ATBM, masyarakat perajin Gamplong mampu menghasilkan kain tenun sebagai bahan stagen (kain panjang untuk melilit bagian perut wanita). Selain kerajinan tenun, perajin juga mampu memproduksi kerajinan anyaman untuk suvenir.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 416, Estimated Rating: 4.15
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40


**Top Recommendations for User 35**
- Recommended Category: 416, Estimated Rating: 4.01
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 139, Estimated Rating: 3.88
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.85
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.85
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.82
- Places in category '97' with similar rating ---
- - No similar places found


**Top Recommendations for User 36**
- Recommended Category: 416, Estimated Rating: 4.35
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 279, Estimated Rating: 4.12
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 4.06
- Places in category '132' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 4.04
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 4.02
- Places in category '28' with similar rating ---
- - No similar places found


**Top Recommendations for User 37**
- Recommended Category: 416, Estimated Rating: 4.44
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 254, Estimated Rating: 4.19
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 157, Estimated Rating: 4.18
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 1, Estimated Rating: 4.02
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 146, Estimated Rating: 4.01
- Places in category '146' with similar rating ---
- - No similar places found


**Top Recommendations for User 38**
- Recommended Category: 322, Estimated Rating: 4.38
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 416, Estimated Rating: 4.32
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 157, Estimated Rating: 4.31
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 112, Estimated Rating: 4.18
- Places in category '112' with similar rating ---
- Place: 112
- Place: Bukit Bintang Yogyakarta
- Description: Bukit Bintang merupakan salah satu lokasi nongkrong favorit di Yogyakarta. Saat malam tiba, pemandangan Yogyakarta sangatlah indah!Terletak di perbatasan Bantul dan Gunungkidul, siapa pun yang berkunjung ke kawasan ini dapat menikmati taburan gemintang di langit malam serta kerlip benderang lampu kota dari ketinggian.Kota ini memiliki Bukit Bintang yang selalu ramai dipadati kawula muda untuk menikmati keindahan malam. Bukit Bintang menjadi tempat yang sempurna untuk menikmati senja hingga malam tiba._x000D_
- City: Yogyakarta
- Price: 25000
- Rating: 4.50

- Recommended Category: 28, Estimated Rating: 4.15
- Places in category '28' with similar rating ---
- Place: 28
- Place: Wisata Agro Edukatif Istana Susu Cibugary
- Description: Kawasan Wisata Agro Edukatif Istana Susu ‚ÄúCibugary‚Äù (Cibubur Garden Dairy ) merupakan suatu kawasan agro peternakan sapi perah bernuansa kebun dan taman. Wisata edukasi Cibubur Garden Dairy (CIBUGARY) merupakan salah wisata edukasi yang diminati siswa sekolah. Edukasi pengenalan peternakan sapi memberikan pengalaman yang tidak dapat dilupakan oleh siswa sekolah. Kegiatan edukasi yang tidak terlalu banyak dan menyenangkan ini sangat disukai siswa sekolah. Dengan berwisata edukasi ini dapat menambah ilmu dan pengalaman secara langsung bagi siswa sekolah. Paket wisata edukatif Istana Susu Cibugary dihargai antara Rp 35.000 sampai Rp 65.000 per orang. Harga tersebut sudah termasuk beberapa botol susu pasteurisasi dan goodybag cibugary.
- City: Jakarta
- Price: 35000
- Rating: 4.50


**Top Recommendations for User 39**
- Recommended Category: 157, Estimated Rating: 4.43
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 97, Estimated Rating: 4.35
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 112, Estimated Rating: 4.34
- Places in category '112' with similar rating ---
- Place: 112
- Place: Bukit Bintang Yogyakarta
- Description: Bukit Bintang merupakan salah satu lokasi nongkrong favorit di Yogyakarta. Saat malam tiba, pemandangan Yogyakarta sangatlah indah!Terletak di perbatasan Bantul dan Gunungkidul, siapa pun yang berkunjung ke kawasan ini dapat menikmati taburan gemintang di langit malam serta kerlip benderang lampu kota dari ketinggian.Kota ini memiliki Bukit Bintang yang selalu ramai dipadati kawula muda untuk menikmati keindahan malam. Bukit Bintang menjadi tempat yang sempurna untuk menikmati senja hingga malam tiba._x000D_
- City: Yogyakarta
- Price: 25000
- Rating: 4.50

- Recommended Category: 134, Estimated Rating: 4.26
- Places in category '134' with similar rating ---
- Place: 134
- Place: Desa Wisata Gamplong
- Description: Desa Wisata Gamplong adalah desa wisata kerajinan tenun yang berada di Padukuhan Gamplong Desa Sumber Rahayu Kecamatan Moyudan Kabupaten Sleman, Yogyakarta. Desa wisata yang terletak di sebelah barat Kota Yogyakarta ini cukup menarik untuk disinggahi wisatawan terlebih karena masih adanya industri kerajinan tenun tradisional dengan menggunakan Alat Tenun Bukan Mesin (ATBM). Dengan ATBM, masyarakat perajin Gamplong mampu menghasilkan kain tenun sebagai bahan stagen (kain panjang untuk melilit bagian perut wanita). Selain kerajinan tenun, perajin juga mampu memproduksi kerajinan anyaman untuk suvenir.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 416, Estimated Rating: 4.22
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40


**Top Recommendations for User 40**
- Recommended Category: 396, Estimated Rating: 4.20
- Places in category '396' with similar rating ---
- Place: 396
- Place: Monumen Kapal Selam
- Description: Monumen Kapal Selam, atau disingkat Monkasel, adalah sebuah museum kapal selam yang terdapat di Embong Kaliasin, Genteng, Surabaya. Terletak di pusat kota, monumen ini sebenarnya merupakan kapal selam KRI Pasopati 410, salah satu armada Angkatan Laut Republik Indonesia buatan Uni Soviet tahun 1952. Kapal selam ini pernah dilibatkan dalam Pertempuran Laut Aru untuk membebaskan Irian Barat dari pendudukan Belanda.\nKapal selam ini kemudian dibawa ke darat dan dijadikan monumen untuk memperingati keberanian pahlawan Indonesia. Monkasel berada di Jalan Pemuda, tepat di sebelah Plaza Surabaya. Selain interior kapal selam, di sini juga diadakan pemutaran film tentang proses peperangan yang terjadi di Laut Aru. Jika ingin mengunjungi tempat wisata ini, maka akan ditemani oleh seorang pemandu lokal yang terdapat di sana.\nAda cerita unik di balik hadirnya monumen Kapal Selam ini. Pada suatu malam Pak Drajat Budiyanto yang merupakan mantan KKM KRI Pasopati 410 (buatan Rusia) ini dan juga mantan KKM KRI Cakra 401 (buatan Jerman Barat), bermimpi diperintahkan oleh KSAL pada waktu itu untuk membawa kapal selam ini melayari Kali Mas. Ternyata mimpi itu menjadi kenyataan. Dia ditugaskan untuk memajang kapal selam di samping Surabaya Plaza. Caranya dengan memotong kapal selam ini menjadi beberapa bagian, kemudian diangkut ke darat, dan dirangkai dan disambung kembali menjadi kapal selam yang utuh.
- City: Surabaya
- Price: 15000
- Rating: 4.40

- Recommended Category: 157, Estimated Rating: 4.15
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 251, Estimated Rating: 4.09
- Places in category '251' with similar rating ---
- Place: 251
- Place: Taman Lansia
- Description: Berlibur santai di akhir pekan cocok dilakukan dengan menghabiskan waktu di taman. Salah satu taman yang dapat menjadi tujuan wisata adalah Taman Lansia Bandung. Lansia merupakan singkatan dari Lanjut Usia. Meski begitu, taman ini tidak dikhususkan untuk para lansia, namun untuk semua kalangan. Nama lansia kemungkinan diberikan karena usia taman ini yang sudah sangat tua. Bahkan, usianya sudah ratusan tahun, karena sudah ada sejak tahun 1885.
- City: Bandung
- Price: 0
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.08
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 139, Estimated Rating: 4.08
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 41**
- Recommended Category: 416, Estimated Rating: 3.90
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.83
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.82
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.81
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 3.81
- Places in category '422' with similar rating ---
- - No similar places found


**Top Recommendations for User 42**
- Recommended Category: 157, Estimated Rating: 4.20
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 399, Estimated Rating: 4.13
- Places in category '399' with similar rating ---
- Place: 399
- Place: Taman Pelangi
- Description: Kalau pelangi biasanya ada di siang hari pasca hujan, maka di Taman Pelangi Yogyakarta pengunjung justru bisa menikmatinya setiap malam hari. Ya, ini lantaran taman ini merupakan Taman Lampion beraneka warna dan rupa. Buka sejak petang, Taman ini memang sangat cocok dijadikan pilihan destinasi rekreasi keluarga saat malam hari. Namun karena merupakan taman outdoor alias terbuka, maka pilihan paling tepat untuk datang ialah saat cuaca tidak hujan.
- City: Surabaya
- Price: 0
- Rating: 4.50

- Recommended Category: 416, Estimated Rating: 4.12
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 321, Estimated Rating: 4.11
- Places in category '321' with similar rating ---
- Place: 321
- Place: Glamping Lakeside Rancabali
- Description: Glamping Lakeside Rancabali menawarkan tempat berkemah mewah yang cocok untuk Anda jadikan destinasi liburan di akhir pekan. Seperti namanya, glamping atau glamour camping terdiri dari tenda yang berisikan tempat tidur nyaman, kamar mandi di dalam ruangan, pemanas, dan berbagai amenitas layaknya hotel berbintang. Berlokasi di Jalan Raya Ciwidey Rancabali KM 1, Glamping Lakeside berada di sekitar Situ Patenggang dan perkebunan teh yang subur. Tidak heran jika pemandangan dari glamping ini begitu apik dan Instagramable untuk berfoto.
- City: Bandung
- Price: 30000
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.08
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20


**Top Recommendations for User 43**
- Recommended Category: 157, Estimated Rating: 4.26
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 28, Estimated Rating: 4.24
- Places in category '28' with similar rating ---
- Place: 28
- Place: Wisata Agro Edukatif Istana Susu Cibugary
- Description: Kawasan Wisata Agro Edukatif Istana Susu ‚ÄúCibugary‚Äù (Cibubur Garden Dairy ) merupakan suatu kawasan agro peternakan sapi perah bernuansa kebun dan taman. Wisata edukasi Cibubur Garden Dairy (CIBUGARY) merupakan salah wisata edukasi yang diminati siswa sekolah. Edukasi pengenalan peternakan sapi memberikan pengalaman yang tidak dapat dilupakan oleh siswa sekolah. Kegiatan edukasi yang tidak terlalu banyak dan menyenangkan ini sangat disukai siswa sekolah. Dengan berwisata edukasi ini dapat menambah ilmu dan pengalaman secara langsung bagi siswa sekolah. Paket wisata edukatif Istana Susu Cibugary dihargai antara Rp 35.000 sampai Rp 65.000 per orang. Harga tersebut sudah termasuk beberapa botol susu pasteurisasi dan goodybag cibugary.
- City: Jakarta
- Price: 35000
- Rating: 4.50

- Recommended Category: 254, Estimated Rating: 4.23
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 416, Estimated Rating: 4.23
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 224, Estimated Rating: 4.12
- Places in category '224' with similar rating ---
- Place: 224
- Place: Dago Dreampark
- Description: Dago Dreampark merupakan wisata kekinian di Kota Bandung dengan luas 13 hektar yang mengusung konsep Jawa - Sunda & Bali dengan dilengkapi berbagai fasilitas & wahana yang menarik.
- City: Bandung
- Price: 40000
- Rating: 4.20


**Top Recommendations for User 44**
- Recommended Category: 1, Estimated Rating: 4.19
- Places in category '1' with similar rating ---
- Place: 1
- Place: Monumen Nasional
- Description: Monumen Nasional atau yang populer disingkat dengan Monas atau Tugu Monas adalah monumen peringatan setinggi 132 meter (433 kaki) yang didirikan untuk mengenang perlawanan dan perjuangan rakyat Indonesia untuk merebut kemerdekaan dari pemerintahan kolonial Hindia Belanda. Pembangunan monumen ini dimulai pada tanggal 17 Agustus 1961 di bawah perintah presiden Soekarno dan dibuka untuk umum pada tanggal 12 Juli 1975. Tugu ini dimahkotai lidah api yang dilapisi lembaran emas yang melambangkan semangat perjuangan yang menyala-nyala. Monumen Nasional terletak tepat di tengah Lapangan Medan Merdeka, Jakarta Pusat.
- City: Jakarta
- Price: 20000
- Rating: 4.60

- Recommended Category: 416, Estimated Rating: 4.12
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 52, Estimated Rating: 4.03
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 4.02
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 4.02
- Places in category '332' with similar rating ---
- - No similar places found


**Top Recommendations for User 45**
- Recommended Category: 139, Estimated Rating: 4.35
- Places in category '139' with similar rating ---
- Place: 139
- Place: Puncak Gunung Api Purba - Nglanggeran
- Description: Gunung Nglanggeran adalah sebuah gunung di Daerah Istimewa Yogyakarta, Indonesia. Gunung ini merupakan suatu gunung api purba yang terbentuk sekitar 0,6-70 juta tahun yang lalu atau yang memiliki umur tersier (Oligo-Miosen). Gunung Nglanggeran memiliki batuan yang sangat khas karena didominasi oleh aglomerat dan breksi gunung api. Gunung ini terletak di Desa Nglanggeran, Kecamatan Patuk, Kabupaten Gunung Kidul yang berada pada deretan Pegunungan Baturagung.
- City: Yogyakarta
- Price: 10000
- Rating: 4.70

- Recommended Category: 322, Estimated Rating: 4.33
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 52, Estimated Rating: 4.31
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 224, Estimated Rating: 4.24
- Places in category '224' with similar rating ---
- Place: 224
- Place: Dago Dreampark
- Description: Dago Dreampark merupakan wisata kekinian di Kota Bandung dengan luas 13 hektar yang mengusung konsep Jawa - Sunda & Bali dengan dilengkapi berbagai fasilitas & wahana yang menarik.
- City: Bandung
- Price: 40000
- Rating: 4.20

- Recommended Category: 134, Estimated Rating: 4.22
- Places in category '134' with similar rating ---
- Place: 134
- Place: Desa Wisata Gamplong
- Description: Desa Wisata Gamplong adalah desa wisata kerajinan tenun yang berada di Padukuhan Gamplong Desa Sumber Rahayu Kecamatan Moyudan Kabupaten Sleman, Yogyakarta. Desa wisata yang terletak di sebelah barat Kota Yogyakarta ini cukup menarik untuk disinggahi wisatawan terlebih karena masih adanya industri kerajinan tenun tradisional dengan menggunakan Alat Tenun Bukan Mesin (ATBM). Dengan ATBM, masyarakat perajin Gamplong mampu menghasilkan kain tenun sebagai bahan stagen (kain panjang untuk melilit bagian perut wanita). Selain kerajinan tenun, perajin juga mampu memproduksi kerajinan anyaman untuk suvenir.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40


**Top Recommendations for User 46**
- Recommended Category: 416, Estimated Rating: 4.65
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.54
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 146, Estimated Rating: 4.40
- Places in category '146' with similar rating ---
- Place: 146
- Place: Bukit Wisata Pulepayung
- Description: Pule Payung Yogyakarta. Sebuah objek wisata populer yang menyajikan keindahan alam dari ketinggian 500 mdpl. Bukit Wisata Pule Payung Yogyakarta merupakan kawasan wisata yang dipenuhi dengan kreativitas yang tinggi, merubah zona biasa menjadi luar biasa. Daya tarik yang pertama dari Pule Payung Yogyakarta adalah objek wisata yang menyajikan pesona alam pegunungan. Sehingga, orang-orang menyebutnya dengan Bukit Wisata. Ketinggian Pule Payung Yogyakarta sekitar 500 mdpl, serta jalan menuju Pule Payung Yogyakarta dapat ditempuh dengan kendaraan roda dua, maupun roda empat. Pesona keindahan Waduk Sermo dapat terlihat jelas saat berada di Pule Payung. Waduk eksotis yang dikelilingi pegunungan yang masih asri. Dan Pule Payung Yogyakarta adalah bukit yang notabene sebagai destinasi wisata alam, diberikan sentuhan kreativitas, serta memprioritaskan aspek keamanan di setiap spotnya, akhirnya menjadi wisata favorit di Jogja.
- City: Yogyakarta
- Price: 10000
- Rating: 4.50

- Recommended Category: 28, Estimated Rating: 4.37
- Places in category '28' with similar rating ---
- Place: 28
- Place: Wisata Agro Edukatif Istana Susu Cibugary
- Description: Kawasan Wisata Agro Edukatif Istana Susu ‚ÄúCibugary‚Äù (Cibubur Garden Dairy ) merupakan suatu kawasan agro peternakan sapi perah bernuansa kebun dan taman. Wisata edukasi Cibubur Garden Dairy (CIBUGARY) merupakan salah wisata edukasi yang diminati siswa sekolah. Edukasi pengenalan peternakan sapi memberikan pengalaman yang tidak dapat dilupakan oleh siswa sekolah. Kegiatan edukasi yang tidak terlalu banyak dan menyenangkan ini sangat disukai siswa sekolah. Dengan berwisata edukasi ini dapat menambah ilmu dan pengalaman secara langsung bagi siswa sekolah. Paket wisata edukatif Istana Susu Cibugary dihargai antara Rp 35.000 sampai Rp 65.000 per orang. Harga tersebut sudah termasuk beberapa botol susu pasteurisasi dan goodybag cibugary.
- City: Jakarta
- Price: 35000
- Rating: 4.50

- Recommended Category: 183, Estimated Rating: 4.28
- Places in category '183' with similar rating ---
- Place: 183
- Place: Jogja Bay Pirates Adventure Waterpark
- Description: Jogja Bay Waterpark atau Jogja Bay (bahasa Jawa: Í¶†Í¶ºÍ¶≠Í¶∏Í¶èÍßÄÍ¶îÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ 'Teluk Ngayogya') adalah salah satu taman wisata air atau waterpark terbesar di Asia Tenggara yang berlokasi di Yogyakarta, Indonesia. Taman wisata air ini dibuka secara resmi pada 20 Desember 2015.
- City: Yogyakarta
- Price: 150000
- Rating: 4.40


**Top Recommendations for User 47**
- Recommended Category: 416, Estimated Rating: 3.61
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 399, Estimated Rating: 3.29
- Places in category '399' with similar rating ---
- - No similar places found

- Recommended Category: 60, Estimated Rating: 3.22
- Places in category '60' with similar rating ---
- - No similar places found

- Recommended Category: 5, Estimated Rating: 3.21
- Places in category '5' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.21
- Places in category '300' with similar rating ---
- - No similar places found


**Top Recommendations for User 48**
- Recommended Category: 254, Estimated Rating: 3.72
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.72
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.69
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.61
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.61
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 49**
- Recommended Category: 416, Estimated Rating: 3.97
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.96
- Places in category '314' with similar rating ---
- Place: 314
- Place: Tafso Barn
- Description: Nama Punclut mungkin sudah cukup akrab di telinga wisatawan. Kawasan dataran tinggi di Bandung yang belakangan populer sebagai destinasi wisata. Selain menawarkan pemandangan yang indah, kawasan ini juga memiliki banyak tempat wisata untuk rekreasi. Salah satunya adalah kawasan wisata sekaligus restoran Tafso dan Boda Barn atau sering dikenal juga dengan Tafso Barn. Tempat rekreasi ini memadukan tempat makan dengan nuansa taman dan gudang yang unik. Pemandangan perbukitan yang dipotong garis horison menjadi incaran utama pengunjung. Sangat tepat nongkrong dan bersantai ketika berlibur di Bandung.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 1, Estimated Rating: 3.89
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 308, Estimated Rating: 3.65
- Places in category '308' with similar rating ---
- - No similar places found

- Recommended Category: 136, Estimated Rating: 3.58
- Places in category '136' with similar rating ---
- - No similar places found


**Top Recommendations for User 50**
- Recommended Category: 416, Estimated Rating: 3.73
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.62
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 3.56
- Places in category '83' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.54
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 3.53
- Places in category '422' with similar rating ---
- - No similar places found


**Top Recommendations for User 51**
- Recommended Category: 416, Estimated Rating: 3.82
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 230, Estimated Rating: 3.76
- Places in category '230' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.72
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.72
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.70
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 52**
- Recommended Category: 416, Estimated Rating: 4.18
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 208, Estimated Rating: 4.02
- Places in category '208' with similar rating ---
- - No similar places found

- Recommended Category: 146, Estimated Rating: 3.93
- Places in category '146' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.93
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 167, Estimated Rating: 3.90
- Places in category '167' with similar rating ---
- - No similar places found


**Top Recommendations for User 53**
- Recommended Category: 52, Estimated Rating: 3.81
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.72
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.71
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 91, Estimated Rating: 3.68
- Places in category '91' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.68
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 54**
- Recommended Category: 254, Estimated Rating: 3.89
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.77
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 136, Estimated Rating: 3.73
- Places in category '136' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.66
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.64
- Places in category '314' with similar rating ---
- - No similar places found


**Top Recommendations for User 55**
- Recommended Category: 416, Estimated Rating: 3.82
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.67
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.67
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.65
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.60
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 56**
- Recommended Category: 416, Estimated Rating: 3.77
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.58
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.46
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.46
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 263, Estimated Rating: 3.40
- Places in category '263' with similar rating ---
- - No similar places found


**Top Recommendations for User 57**
- Recommended Category: 139, Estimated Rating: 3.53
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.39
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.33
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.31
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 249, Estimated Rating: 3.31
- Places in category '249' with similar rating ---
- - No similar places found


**Top Recommendations for User 58**
- Recommended Category: 416, Estimated Rating: 3.75
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.73
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.60
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 340, Estimated Rating: 3.56
- Places in category '340' with similar rating ---
- Place: 340
- Place: Desa Wisata Lembah Kalipancur
- Description: Wisata alam tengah menjadi sorotan bagi dunia pariwisata. Wisata alam tidak hanya pegunungan, danau, hutan, air terjun, atau pantai. Tapi juga merambah ke konsep pedesaan, seperti Desa Wisata Lembah Kalipancur. Tempat wisata ini menawarkan rekreasi berkonsep pedesaan yang ada di Kota Semarang. Meskipun berkonsep pedesaan, lokasinya tidak jauh dari pusat kota. Dan objek wisata yang sudah lama berdiri ini, masih tetap diminati oleh wisatawan.
- City: Semarang
- Price: 0
- Rating: 3.90

- Recommended Category: 44, Estimated Rating: 3.55
- Places in category '44' with similar rating ---
- - No similar places found


**Top Recommendations for User 59**
- Recommended Category: 416, Estimated Rating: 4.01
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 139, Estimated Rating: 3.87
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.85
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.83
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.81
- Places in category '1' with similar rating ---
- - No similar places found


**Top Recommendations for User 60**
- Recommended Category: 416, Estimated Rating: 3.86
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 224, Estimated Rating: 3.73
- Places in category '224' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.71
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.70
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.69
- Places in category '322' with similar rating ---
- - No similar places found


**Top Recommendations for User 61**
- Recommended Category: 407, Estimated Rating: 4.26
- Places in category '407' with similar rating ---
- Place: 407
- Place: Taman Ekspresi Dan Perpustakaan
- Description: Taman Ekspresi Surabaya tidak hanya menyuguhkan taman yang rindang dan asri. Taman di Jalan Gentengkali ini juga punya fasilitas penunjang berupa perpustakaan di sisi barat. Perpustakaan yang dikelola Dinas Perpustakaan dan Kearsipan Kota Surabaya tersebut memiliki 3.412 koleksi buku. Buku-buku yang ada di perpustakaan sangat lengkap dan baru, dari cetakan 2009 sampai terbaru 2019. Mulai buku bahasa, filsafat, psikologi, agama, matematika, ilmu alam, ilmu sosial, keluarga, teknologi, seni, sastra, geografis, sejarah, hingga bacaan untuk anak-anak.
- City: Surabaya
- Price: 0
- Rating: 4.50

- Recommended Category: 387, Estimated Rating: 4.11
- Places in category '387' with similar rating ---
- Place: 387
- Place: Obyek Wisata Goa Kreo
- Description: Goa Kreo Semarang yang berada di ibukota Jawa Tengah ini begitu hit di kalangan netizen. Tak heran jika cuaca cerah, antrian menuju ke sana begitu mengular. Meski tak ada angkutan umum menuju ke sana, tak menghalangi rasa penasaran wisatawan. Gua ini terbentuk secara alami yang membedakan gua ini dengan gua lainnya adalah letaknya. lokasinya terletak di tengah waduk jatibarang, sebuah bendungan yang membendung Sungai Kreo.
- City: Semarang
- Price: 5500
- Rating: 4.30

- Recommended Category: 246, Estimated Rating: 4.09
- Places in category '246' with similar rating ---
- Place: 246
- Place: Curug Tilu Leuwi Opat
- Description: Curug Tilu Leuwi Opat merupakan salah satu wisata curug di Lembang. Tempatnya sendiri cukup luas. Disini ada area outbond, camping, dan tentunya wisata air terjun dan sungai. Area depan berupa lembah dengan sungai jernih. Biasanya outbond, camping, dan permainan dilakukan di area ini. Lokasi curug tilu leuwi opat sebenarnya bertetangga langsung dengan Dusun Bambu lho. Pernah ke dusun bambu? jika naik ke skywalk lutung kasarung, atau mengunjungi area camping, anda bisa melihat lembah yang berbatasan langsung dengan dusun bambu. Nah, lembah itu termasuk ke dalam area wisata curug tilu leuwi opat. Di lembah ini mengalir sungai dengan air jernih yang bersumber dari Situ Lembang.
- City: Bandung
- Price: 10000
- Rating: 4.40

- Recommended Category: 416, Estimated Rating: 4.09
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 321, Estimated Rating: 4.07
- Places in category '321' with similar rating ---
- Place: 321
- Place: Glamping Lakeside Rancabali
- Description: Glamping Lakeside Rancabali menawarkan tempat berkemah mewah yang cocok untuk Anda jadikan destinasi liburan di akhir pekan. Seperti namanya, glamping atau glamour camping terdiri dari tenda yang berisikan tempat tidur nyaman, kamar mandi di dalam ruangan, pemanas, dan berbagai amenitas layaknya hotel berbintang. Berlokasi di Jalan Raya Ciwidey Rancabali KM 1, Glamping Lakeside berada di sekitar Situ Patenggang dan perkebunan teh yang subur. Tidak heran jika pemandangan dari glamping ini begitu apik dan Instagramable untuk berfoto.
- City: Bandung
- Price: 30000
- Rating: 4.40


**Top Recommendations for User 62**
- Recommended Category: 322, Estimated Rating: 4.19
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 416, Estimated Rating: 4.11
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 91, Estimated Rating: 3.96
- Places in category '91' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.92
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 164, Estimated Rating: 3.91
- Places in category '164' with similar rating ---
- - No similar places found


**Top Recommendations for User 63**
- Recommended Category: 157, Estimated Rating: 3.71
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.59
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.57
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.49
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 136, Estimated Rating: 3.49
- Places in category '136' with similar rating ---
- - No similar places found


**Top Recommendations for User 64**
- Recommended Category: 139, Estimated Rating: 3.68
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.66
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.64
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.61
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.59
- Places in category '254' with similar rating ---
- - No similar places found


**Top Recommendations for User 65**
- Recommended Category: 322, Estimated Rating: 3.98
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 53, Estimated Rating: 3.88
- Places in category '53' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.85
- Places in category '132' with similar rating ---
- - No similar places found

- Recommended Category: 146, Estimated Rating: 3.85
- Places in category '146' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.81
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 66**
- Recommended Category: 139, Estimated Rating: 3.48
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 167, Estimated Rating: 3.30
- Places in category '167' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 3.28
- Places in category '401' with similar rating ---
- - No similar places found

- Recommended Category: 91, Estimated Rating: 3.26
- Places in category '91' with similar rating ---
- - No similar places found

- Recommended Category: 253, Estimated Rating: 3.25
- Places in category '253' with similar rating ---
- - No similar places found


**Top Recommendations for User 67**
- Recommended Category: 322, Estimated Rating: 3.60
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.58
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.52
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.52
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 230, Estimated Rating: 3.51
- Places in category '230' with similar rating ---
- - No similar places found


**Top Recommendations for User 68**
- Recommended Category: 416, Estimated Rating: 4.13
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 422, Estimated Rating: 4.00
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 333, Estimated Rating: 3.97
- Places in category '333' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.93
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.88
- Places in category '254' with similar rating ---
- - No similar places found


**Top Recommendations for User 69**
- Recommended Category: 416, Estimated Rating: 3.42
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.32
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.31
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.26
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.22
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 70**
- Recommended Category: 416, Estimated Rating: 4.24
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 139, Estimated Rating: 4.13
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 4.03
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 4.00
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.96
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 71**
- Recommended Category: 97, Estimated Rating: 3.78
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.66
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.65
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.64
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.60
- Places in category '416' with similar rating ---
- - No similar places found


**Top Recommendations for User 72**
- Recommended Category: 416, Estimated Rating: 3.77
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.73
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.58
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.57
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.56
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 73**
- Recommended Category: 322, Estimated Rating: 3.76
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.52
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.49
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.38
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.34
- Places in category '112' with similar rating ---
- - No similar places found


**Top Recommendations for User 74**
- Recommended Category: 146, Estimated Rating: 3.72
- Places in category '146' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.71
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.70
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 53, Estimated Rating: 3.65
- Places in category '53' with similar rating ---
- - No similar places found

- Recommended Category: 224, Estimated Rating: 3.64
- Places in category '224' with similar rating ---
- - No similar places found


**Top Recommendations for User 75**
- Recommended Category: 416, Estimated Rating: 4.12
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.08
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 112, Estimated Rating: 4.05
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 90, Estimated Rating: 4.01
- Places in category '90' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 4.01
- Places in category '401' with similar rating ---
- Place: 401
- Place: Taman Keputran
- Description: Ntah, mengapa nama taman ini disebut dengan taman keputran, namun jika dikaitkan dengan lokasi taman, berada di persimpangan jalan Keputran Surabaya. Selain itu, di seberang sungai terdapat juga sebuah taman yang bernama Taman Lalu Lintas Surabaya. Keunikan Taman Keputran dibandingkan dengan taman lainnya adalah adanya lampu yang kerangka tiangnya mirip dengan manusia, dengan gaya pose yang berbeda-beda itu, membuat taman ini terlihat ada sentuhan kreatif dan inovatif. Lampu-lampu itu terang dan indah pada malam hari. Untuknya, taman ini cukup representatif sebagai pilihan mendapatkan tempat tongkrongan yang tenang pada malam hari. Tak hanya itu, jalur pijat refleksi kaki juga tersedia disana. Cocok untuk pengunjung yang ingin mencoba pengobatan refleksi kaki sederhana. Melalui pemahaman tentang jalur refleksi secara mendalam, sebagian orang memilih sarana ini dalam usaha menyembuhkan penyakit._x000D_
- City: Surabaya
- Price: 0
- Rating: 4.30


**Top Recommendations for User 76**
- Recommended Category: 157, Estimated Rating: 3.70
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.47
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 251, Estimated Rating: 3.44
- Places in category '251' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.43
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.41
- Places in category '300' with similar rating ---
- - No similar places found


**Top Recommendations for User 77**
- Recommended Category: 136, Estimated Rating: 3.90
- Places in category '136' with similar rating ---
- - No similar places found

- Recommended Category: 164, Estimated Rating: 3.90
- Places in category '164' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.88
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 134, Estimated Rating: 3.84
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.80
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 78**
- Recommended Category: 97, Estimated Rating: 4.15
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 44, Estimated Rating: 4.10
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 4.10
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.06
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 254, Estimated Rating: 3.99
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30


**Top Recommendations for User 79**
- Recommended Category: 332, Estimated Rating: 3.67
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.53
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.49
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 138, Estimated Rating: 3.49
- Places in category '138' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.47
- Places in category '112' with similar rating ---
- - No similar places found


**Top Recommendations for User 80**
- Recommended Category: 416, Estimated Rating: 3.44
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.35
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.33
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.31
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 387, Estimated Rating: 3.26
- Places in category '387' with similar rating ---
- - No similar places found


**Top Recommendations for User 81**
- Recommended Category: 1, Estimated Rating: 3.53
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 353, Estimated Rating: 3.38
- Places in category '353' with similar rating ---
- - No similar places found

- Recommended Category: 117, Estimated Rating: 3.37
- Places in category '117' with similar rating ---
- - No similar places found

- Recommended Category: 164, Estimated Rating: 3.32
- Places in category '164' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.31
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 82**
- Recommended Category: 416, Estimated Rating: 4.07
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 314, Estimated Rating: 3.87
- Places in category '314' with similar rating ---
- Place: 314
- Place: Tafso Barn
- Description: Nama Punclut mungkin sudah cukup akrab di telinga wisatawan. Kawasan dataran tinggi di Bandung yang belakangan populer sebagai destinasi wisata. Selain menawarkan pemandangan yang indah, kawasan ini juga memiliki banyak tempat wisata untuk rekreasi. Salah satunya adalah kawasan wisata sekaligus restoran Tafso dan Boda Barn atau sering dikenal juga dengan Tafso Barn. Tempat rekreasi ini memadukan tempat makan dengan nuansa taman dan gudang yang unik. Pemandangan perbukitan yang dipotong garis horison menjadi incaran utama pengunjung. Sangat tepat nongkrong dan bersantai ketika berlibur di Bandung.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 1, Estimated Rating: 3.84
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 202, Estimated Rating: 3.75
- Places in category '202' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.70
- Places in category '322' with similar rating ---
- - No similar places found


**Top Recommendations for User 83**
- Recommended Category: 416, Estimated Rating: 3.86
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.84
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.81
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.76
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 142, Estimated Rating: 3.76
- Places in category '142' with similar rating ---
- - No similar places found


**Top Recommendations for User 84**
- Recommended Category: 28, Estimated Rating: 4.10
- Places in category '28' with similar rating ---
- Place: 28
- Place: Wisata Agro Edukatif Istana Susu Cibugary
- Description: Kawasan Wisata Agro Edukatif Istana Susu ‚ÄúCibugary‚Äù (Cibubur Garden Dairy ) merupakan suatu kawasan agro peternakan sapi perah bernuansa kebun dan taman. Wisata edukasi Cibubur Garden Dairy (CIBUGARY) merupakan salah wisata edukasi yang diminati siswa sekolah. Edukasi pengenalan peternakan sapi memberikan pengalaman yang tidak dapat dilupakan oleh siswa sekolah. Kegiatan edukasi yang tidak terlalu banyak dan menyenangkan ini sangat disukai siswa sekolah. Dengan berwisata edukasi ini dapat menambah ilmu dan pengalaman secara langsung bagi siswa sekolah. Paket wisata edukatif Istana Susu Cibugary dihargai antara Rp 35.000 sampai Rp 65.000 per orang. Harga tersebut sudah termasuk beberapa botol susu pasteurisasi dan goodybag cibugary.
- City: Jakarta
- Price: 35000
- Rating: 4.50

- Recommended Category: 232, Estimated Rating: 4.01
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.97
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.92
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.89
- Places in category '321' with similar rating ---
- - No similar places found


**Top Recommendations for User 85**
- Recommended Category: 1, Estimated Rating: 3.59
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.49
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.46
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.45
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.44
- Places in category '97' with similar rating ---
- - No similar places found


**Top Recommendations for User 86**
- Recommended Category: 157, Estimated Rating: 3.80
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.72
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.71
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.68
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.67
- Places in category '416' with similar rating ---
- - No similar places found


**Top Recommendations for User 87**
- Recommended Category: 322, Estimated Rating: 3.72
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.72
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.64
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 138, Estimated Rating: 3.64
- Places in category '138' with similar rating ---
- - No similar places found

- Recommended Category: 125, Estimated Rating: 3.59
- Places in category '125' with similar rating ---
- - No similar places found


**Top Recommendations for User 88**
- Recommended Category: 157, Estimated Rating: 3.08
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.00
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.00
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 224, Estimated Rating: 2.94
- Places in category '224' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 2.89
- Places in category '83' with similar rating ---
- - No similar places found


**Top Recommendations for User 89**
- Recommended Category: 416, Estimated Rating: 3.90
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.83
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.70
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.70
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.66
- Places in category '1' with similar rating ---
- - No similar places found


**Top Recommendations for User 90**
- Recommended Category: 416, Estimated Rating: 3.53
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.45
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 94, Estimated Rating: 3.36
- Places in category '94' with similar rating ---
- - No similar places found

- Recommended Category: 200, Estimated Rating: 3.33
- Places in category '200' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 3.29
- Places in category '79' with similar rating ---
- - No similar places found


**Top Recommendations for User 91**
- Recommended Category: 416, Estimated Rating: 4.28
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 138, Estimated Rating: 3.92
- Places in category '138' with similar rating ---
- - No similar places found

- Recommended Category: 246, Estimated Rating: 3.82
- Places in category '246' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.80
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 53, Estimated Rating: 3.80
- Places in category '53' with similar rating ---
- - No similar places found


**Top Recommendations for User 92**
- Recommended Category: 52, Estimated Rating: 4.35
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 416, Estimated Rating: 4.35
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 112, Estimated Rating: 4.22
- Places in category '112' with similar rating ---
- Place: 112
- Place: Bukit Bintang Yogyakarta
- Description: Bukit Bintang merupakan salah satu lokasi nongkrong favorit di Yogyakarta. Saat malam tiba, pemandangan Yogyakarta sangatlah indah!Terletak di perbatasan Bantul dan Gunungkidul, siapa pun yang berkunjung ke kawasan ini dapat menikmati taburan gemintang di langit malam serta kerlip benderang lampu kota dari ketinggian.Kota ini memiliki Bukit Bintang yang selalu ramai dipadati kawula muda untuk menikmati keindahan malam. Bukit Bintang menjadi tempat yang sempurna untuk menikmati senja hingga malam tiba._x000D_
- City: Yogyakarta
- Price: 25000
- Rating: 4.50

- Recommended Category: 157, Estimated Rating: 4.15
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 321, Estimated Rating: 4.08
- Places in category '321' with similar rating ---
- Place: 321
- Place: Glamping Lakeside Rancabali
- Description: Glamping Lakeside Rancabali menawarkan tempat berkemah mewah yang cocok untuk Anda jadikan destinasi liburan di akhir pekan. Seperti namanya, glamping atau glamour camping terdiri dari tenda yang berisikan tempat tidur nyaman, kamar mandi di dalam ruangan, pemanas, dan berbagai amenitas layaknya hotel berbintang. Berlokasi di Jalan Raya Ciwidey Rancabali KM 1, Glamping Lakeside berada di sekitar Situ Patenggang dan perkebunan teh yang subur. Tidak heran jika pemandangan dari glamping ini begitu apik dan Instagramable untuk berfoto.
- City: Bandung
- Price: 30000
- Rating: 4.40


**Top Recommendations for User 93**
- Recommended Category: 416, Estimated Rating: 3.98
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 138, Estimated Rating: 3.98
- Places in category '138' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.88
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 254, Estimated Rating: 3.87
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 318, Estimated Rating: 3.83
- Places in category '318' with similar rating ---
- - No similar places found


**Top Recommendations for User 94**
- Recommended Category: 314, Estimated Rating: 3.78
- Places in category '314' with similar rating ---
- - No similar places found

- Recommended Category: 295, Estimated Rating: 3.75
- Places in category '295' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.74
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.73
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 3.70
- Places in category '422' with similar rating ---
- - No similar places found


**Top Recommendations for User 95**
- Recommended Category: 416, Estimated Rating: 4.32
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 333, Estimated Rating: 4.14
- Places in category '333' with similar rating ---
- Place: 333
- Place: Kota Mini
- Description: Destinasi yang sangat menarik bernuansa eropa lengkap dengan fasilitas publiknya membuat anda seolah-olah sedang berada di eropa. Ada rumah sakit, kantor polisi, box telepon, cafe-caf√© pinggir jalan, dan bangunan-bangunan klasik lainnya. Masih satu kompleks dengan Floating Market, tempat wisata ini menyajikan arsitektur eropa klasik yang begitu kental, dijamin anda akan enggan beranjak.
- City: Bandung
- Price: 20000
- Rating: 4.40

- Recommended Category: 97, Estimated Rating: 4.14
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 139, Estimated Rating: 4.12
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 4.06
- Places in category '1' with similar rating ---
- - No similar places found


**Top Recommendations for User 96**
- Recommended Category: 322, Estimated Rating: 3.64
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.55
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 100, Estimated Rating: 3.42
- Places in category '100' with similar rating ---
- - No similar places found

- Recommended Category: 230, Estimated Rating: 3.42
- Places in category '230' with similar rating ---
- - No similar places found

- Recommended Category: 136, Estimated Rating: 3.41
- Places in category '136' with similar rating ---
- - No similar places found


**Top Recommendations for User 97**
- Recommended Category: 322, Estimated Rating: 4.09
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 416, Estimated Rating: 3.99
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.78
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.77
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 224, Estimated Rating: 3.75
- Places in category '224' with similar rating ---
- - No similar places found


**Top Recommendations for User 98**
- Recommended Category: 300, Estimated Rating: 3.89
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.77
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 387, Estimated Rating: 3.70
- Places in category '387' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.68
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.67
- Places in category '97' with similar rating ---
- - No similar places found


**Top Recommendations for User 99**
- Recommended Category: 416, Estimated Rating: 4.16
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 279, Estimated Rating: 4.03
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 4.01
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 164, Estimated Rating: 3.98
- Places in category '164' with similar rating ---
- - No similar places found

- Recommended Category: 80, Estimated Rating: 3.97
- Places in category '80' with similar rating ---
- - No similar places found


**Top Recommendations for User 100**
- Recommended Category: 416, Estimated Rating: 3.67
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.58
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 251, Estimated Rating: 3.40
- Places in category '251' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.37
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 263, Estimated Rating: 3.35
- Places in category '263' with similar rating ---
- - No similar places found


**Top Recommendations for User 101**
- Recommended Category: 97, Estimated Rating: 3.78
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 94, Estimated Rating: 3.75
- Places in category '94' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.71
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.67
- Places in category '314' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.67
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 102**
- Recommended Category: 254, Estimated Rating: 3.70
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.56
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.55
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.50
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.48
- Places in category '332' with similar rating ---
- - No similar places found


**Top Recommendations for User 103**
- Recommended Category: 416, Estimated Rating: 3.98
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.85
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.74
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 117, Estimated Rating: 3.67
- Places in category '117' with similar rating ---
- - No similar places found

- Recommended Category: 172, Estimated Rating: 3.64
- Places in category '172' with similar rating ---
- - No similar places found


**Top Recommendations for User 104**
- Recommended Category: 416, Estimated Rating: 3.70
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.69
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.63
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 196, Estimated Rating: 3.63
- Places in category '196' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.60
- Places in category '321' with similar rating ---
- - No similar places found


**Top Recommendations for User 105**
- Recommended Category: 139, Estimated Rating: 4.06
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 4.01
- Places in category '79' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 4.00
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 333, Estimated Rating: 3.98
- Places in category '333' with similar rating ---
- - No similar places found

- Recommended Category: 200, Estimated Rating: 3.97
- Places in category '200' with similar rating ---
- - No similar places found


**Top Recommendations for User 106**
- Recommended Category: 416, Estimated Rating: 3.97
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.89
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.81
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.79
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.71
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 107**
- Recommended Category: 139, Estimated Rating: 3.92
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.76
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 246, Estimated Rating: 3.65
- Places in category '246' with similar rating ---
- - No similar places found

- Recommended Category: 164, Estimated Rating: 3.64
- Places in category '164' with similar rating ---
- - No similar places found

- Recommended Category: 287, Estimated Rating: 3.63
- Places in category '287' with similar rating ---
- - No similar places found


**Top Recommendations for User 108**
- Recommended Category: 279, Estimated Rating: 4.01
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 4.01
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 253, Estimated Rating: 3.97
- Places in category '253' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.89
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.87
- Places in category '97' with similar rating ---
- - No similar places found


**Top Recommendations for User 109**
- Recommended Category: 52, Estimated Rating: 4.10
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 97, Estimated Rating: 4.01
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.99
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.96
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.93
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 110**
- Recommended Category: 416, Estimated Rating: 3.66
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.61
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.60
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.54
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.54
- Places in category '52' with similar rating ---
- - No similar places found


**Top Recommendations for User 111**
- Recommended Category: 416, Estimated Rating: 4.15
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 136, Estimated Rating: 3.99
- Places in category '136' with similar rating ---
- - No similar places found

- Recommended Category: 399, Estimated Rating: 3.99
- Places in category '399' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.96
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.93
- Places in category '44' with similar rating ---
- - No similar places found


**Top Recommendations for User 112**
- Recommended Category: 416, Estimated Rating: 3.78
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.78
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.72
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.64
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.62
- Places in category '52' with similar rating ---
- - No similar places found


**Top Recommendations for User 113**
- Recommended Category: 157, Estimated Rating: 4.04
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 139, Estimated Rating: 3.95
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.88
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.80
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 309, Estimated Rating: 3.78
- Places in category '309' with similar rating ---
- - No similar places found


**Top Recommendations for User 114**
- Recommended Category: 416, Estimated Rating: 3.88
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.76
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 425, Estimated Rating: 3.72
- Places in category '425' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.66
- Places in category '132' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.65
- Places in category '314' with similar rating ---
- - No similar places found


**Top Recommendations for User 115**
- Recommended Category: 97, Estimated Rating: 4.24
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 279, Estimated Rating: 4.22
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 4.19
- Places in category '300' with similar rating ---
- Place: 300
- Place: Sanghyang Heuleut
- Description: Danau yang satu ini memiliki air jernih bernuansa kehijauan yang berpadu indah dengan tebing batu berukuran besar di sekelilingnya. Di sela-sela bebatuan, kamu bisa menemukan pepohonan dan rerumputan hijau yang menyegarkan mata. Bebatuannya yang tinggi itu bahkan kerap digunakan sebagai tempat para traveler untuk meloncat indah menuju danau purba tersebut. Danau Sanghyang Heuleut memiliki kedalaman sekitar tiga meter. Jadi, sebelum kamu masuk ke dalam danau, pastikan kamu bisa berenang agar tidak tenggelam, atau setidaknya telah memiliki perlengkapan seperti jaket pelampung. _x000D_
- City: Bandung
- Price: 10000
- Rating: 4.40

- Recommended Category: 251, Estimated Rating: 4.08
- Places in category '251' with similar rating ---
- Place: 251
- Place: Taman Lansia
- Description: Berlibur santai di akhir pekan cocok dilakukan dengan menghabiskan waktu di taman. Salah satu taman yang dapat menjadi tujuan wisata adalah Taman Lansia Bandung. Lansia merupakan singkatan dari Lanjut Usia. Meski begitu, taman ini tidak dikhususkan untuk para lansia, namun untuk semua kalangan. Nama lansia kemungkinan diberikan karena usia taman ini yang sudah sangat tua. Bahkan, usianya sudah ratusan tahun, karena sudah ada sejak tahun 1885.
- City: Bandung
- Price: 0
- Rating: 4.40

- Recommended Category: 416, Estimated Rating: 4.08
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40


**Top Recommendations for User 116**
- Recommended Category: 254, Estimated Rating: 3.57
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.45
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.45
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.41
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 24, Estimated Rating: 3.39
- Places in category '24' with similar rating ---
- - No similar places found


**Top Recommendations for User 117**
- Recommended Category: 416, Estimated Rating: 4.04
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 44, Estimated Rating: 3.93
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.73
- Places in category '314' with similar rating ---
- - No similar places found

- Recommended Category: 80, Estimated Rating: 3.72
- Places in category '80' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.72
- Places in category '112' with similar rating ---
- - No similar places found


**Top Recommendations for User 118**
- Recommended Category: 117, Estimated Rating: 3.90
- Places in category '117' with similar rating ---
- Place: 117
- Place: The World Landmarks - Merapi Park Yogyakarta
- Description: Merapi Park merupakan salah satu tempat wisata di Yogyakarta yang terletak di Jalan Kaliurang km 22, Hargobinangun, Kecamatan Pakem, Kabupaten Sleman. Pengoperasian Merapi Park dimulai sejak tanggal 25 Juni 2017. Fasilitas yang tersedia meliputi tempat pengambilan foto
- City: Yogyakarta
- Price: 22000
- Rating: 4.20

- Recommended Category: 139, Estimated Rating: 3.81
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 122, Estimated Rating: 3.74
- Places in category '122' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.72
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.71
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 119**
- Recommended Category: 157, Estimated Rating: 3.53
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.43
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.41
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 3.40
- Places in category '79' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.40
- Places in category '416' with similar rating ---
- - No similar places found


**Top Recommendations for User 120**
- Recommended Category: 279, Estimated Rating: 4.01
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.98
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.93
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.92
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 117, Estimated Rating: 3.91
- Places in category '117' with similar rating ---
- Place: 117
- Place: The World Landmarks - Merapi Park Yogyakarta
- Description: Merapi Park merupakan salah satu tempat wisata di Yogyakarta yang terletak di Jalan Kaliurang km 22, Hargobinangun, Kecamatan Pakem, Kabupaten Sleman. Pengoperasian Merapi Park dimulai sejak tanggal 25 Juni 2017. Fasilitas yang tersedia meliputi tempat pengambilan foto
- City: Yogyakarta
- Price: 22000
- Rating: 4.20


**Top Recommendations for User 121**
- Recommended Category: 157, Estimated Rating: 3.76
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.66
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 90, Estimated Rating: 3.64
- Places in category '90' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.62
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.62
- Places in category '416' with similar rating ---
- - No similar places found


**Top Recommendations for User 122**
- Recommended Category: 416, Estimated Rating: 3.93
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.73
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 60, Estimated Rating: 3.71
- Places in category '60' with similar rating ---
- - No similar places found

- Recommended Category: 36, Estimated Rating: 3.69
- Places in category '36' with similar rating ---
- - No similar places found

- Recommended Category: 387, Estimated Rating: 3.66
- Places in category '387' with similar rating ---
- - No similar places found


**Top Recommendations for User 123**
- Recommended Category: 157, Estimated Rating: 3.08
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.04
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 308, Estimated Rating: 2.94
- Places in category '308' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 2.93
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 2.90
- Places in category '254' with similar rating ---
- - No similar places found


**Top Recommendations for User 124**
- Recommended Category: 416, Estimated Rating: 3.89
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 142, Estimated Rating: 3.70
- Places in category '142' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.69
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.68
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.63
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 125**
- Recommended Category: 157, Estimated Rating: 3.83
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.77
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 241, Estimated Rating: 3.70
- Places in category '241' with similar rating ---
- - No similar places found

- Recommended Category: 214, Estimated Rating: 3.69
- Places in category '214' with similar rating ---
- - No similar places found

- Recommended Category: 249, Estimated Rating: 3.62
- Places in category '249' with similar rating ---
- - No similar places found


**Top Recommendations for User 126**
- Recommended Category: 279, Estimated Rating: 4.14
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 335, Estimated Rating: 4.04
- Places in category '335' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.99
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 1, Estimated Rating: 3.99
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.91
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 127**
- Recommended Category: 134, Estimated Rating: 3.82
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.82
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 374, Estimated Rating: 3.76
- Places in category '374' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.75
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.72
- Places in category '1' with similar rating ---
- - No similar places found


**Top Recommendations for User 128**
- Recommended Category: 157, Estimated Rating: 3.72
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.72
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 3.70
- Places in category '401' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.65
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 3.59
- Places in category '83' with similar rating ---
- - No similar places found


**Top Recommendations for User 129**
- Recommended Category: 279, Estimated Rating: 4.23
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 4.10
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 4.05
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 136, Estimated Rating: 3.99
- Places in category '136' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.99
- Places in category '321' with similar rating ---
- - No similar places found


**Top Recommendations for User 130**
- Recommended Category: 112, Estimated Rating: 4.23
- Places in category '112' with similar rating ---
- Place: 112
- Place: Bukit Bintang Yogyakarta
- Description: Bukit Bintang merupakan salah satu lokasi nongkrong favorit di Yogyakarta. Saat malam tiba, pemandangan Yogyakarta sangatlah indah!Terletak di perbatasan Bantul dan Gunungkidul, siapa pun yang berkunjung ke kawasan ini dapat menikmati taburan gemintang di langit malam serta kerlip benderang lampu kota dari ketinggian.Kota ini memiliki Bukit Bintang yang selalu ramai dipadati kawula muda untuk menikmati keindahan malam. Bukit Bintang menjadi tempat yang sempurna untuk menikmati senja hingga malam tiba._x000D_
- City: Yogyakarta
- Price: 25000
- Rating: 4.50

- Recommended Category: 97, Estimated Rating: 4.21
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 139, Estimated Rating: 4.18
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 4.01
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 401, Estimated Rating: 4.01
- Places in category '401' with similar rating ---
- Place: 401
- Place: Taman Keputran
- Description: Ntah, mengapa nama taman ini disebut dengan taman keputran, namun jika dikaitkan dengan lokasi taman, berada di persimpangan jalan Keputran Surabaya. Selain itu, di seberang sungai terdapat juga sebuah taman yang bernama Taman Lalu Lintas Surabaya. Keunikan Taman Keputran dibandingkan dengan taman lainnya adalah adanya lampu yang kerangka tiangnya mirip dengan manusia, dengan gaya pose yang berbeda-beda itu, membuat taman ini terlihat ada sentuhan kreatif dan inovatif. Lampu-lampu itu terang dan indah pada malam hari. Untuknya, taman ini cukup representatif sebagai pilihan mendapatkan tempat tongkrongan yang tenang pada malam hari. Tak hanya itu, jalur pijat refleksi kaki juga tersedia disana. Cocok untuk pengunjung yang ingin mencoba pengobatan refleksi kaki sederhana. Melalui pemahaman tentang jalur refleksi secara mendalam, sebagian orang memilih sarana ini dalam usaha menyembuhkan penyakit._x000D_
- City: Surabaya
- Price: 0
- Rating: 4.30


**Top Recommendations for User 131**
- Recommended Category: 416, Estimated Rating: 3.80
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 3.61
- Places in category '401' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.60
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.52
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.50
- Places in category '321' with similar rating ---
- - No similar places found


**Top Recommendations for User 132**
- Recommended Category: 97, Estimated Rating: 4.13
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 112, Estimated Rating: 3.73
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.69
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 399, Estimated Rating: 3.62
- Places in category '399' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.60
- Places in category '314' with similar rating ---
- - No similar places found


**Top Recommendations for User 133**
- Recommended Category: 230, Estimated Rating: 3.94
- Places in category '230' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.90
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.77
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 335, Estimated Rating: 3.74
- Places in category '335' with similar rating ---
- - No similar places found

- Recommended Category: 431, Estimated Rating: 3.73
- Places in category '431' with similar rating ---
- - No similar places found


**Top Recommendations for User 134**
- Recommended Category: 416, Estimated Rating: 4.11
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 314, Estimated Rating: 3.80
- Places in category '314' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.73
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.72
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.70
- Places in category '1' with similar rating ---
- - No similar places found


**Top Recommendations for User 135**
- Recommended Category: 254, Estimated Rating: 3.98
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 422, Estimated Rating: 3.81
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 399, Estimated Rating: 3.80
- Places in category '399' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.77
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 387, Estimated Rating: 3.76
- Places in category '387' with similar rating ---
- - No similar places found


**Top Recommendations for User 136**
- Recommended Category: 416, Estimated Rating: 3.52
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 351, Estimated Rating: 3.26
- Places in category '351' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.20
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.19
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 3.19
- Places in category '79' with similar rating ---
- - No similar places found


**Top Recommendations for User 137**
- Recommended Category: 97, Estimated Rating: 4.07
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 138, Estimated Rating: 4.06
- Places in category '138' with similar rating ---
- Place: 138
- Place: Jogja Exotarium
- Description: Di Yogyakarta, tepatnya di Sleman, ada satu tempat wisata edukasi yang patut dikunjungi. Namanya adalah Jogja Exotarium ‚Äî terdengar unik, kan? Namun sebenarnya, tempat ini merupakan taman hewan berskala kecil. Koleksi hewannya beragam dan bisa diajak berinteraksi secara langsung
- City: Yogyakarta
- Price: 20000
- Rating: 4.40

- Recommended Category: 254, Estimated Rating: 3.90
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 91, Estimated Rating: 3.89
- Places in category '91' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.87
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 138**
- Recommended Category: 332, Estimated Rating: 3.84
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.83
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 396, Estimated Rating: 3.71
- Places in category '396' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.70
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.68
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 139**
- Recommended Category: 1, Estimated Rating: 4.01
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.85
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.83
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.78
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 263, Estimated Rating: 3.70
- Places in category '263' with similar rating ---
- - No similar places found


**Top Recommendations for User 140**
- Recommended Category: 322, Estimated Rating: 4.24
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 416, Estimated Rating: 4.05
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 309, Estimated Rating: 3.93
- Places in category '309' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.92
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.89
- Places in category '52' with similar rating ---
- - No similar places found


**Top Recommendations for User 141**
- Recommended Category: 139, Estimated Rating: 4.26
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 142, Estimated Rating: 4.12
- Places in category '142' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.99
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.92
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 90, Estimated Rating: 3.92
- Places in category '90' with similar rating ---
- - No similar places found


**Top Recommendations for User 142**
- Recommended Category: 157, Estimated Rating: 3.66
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.57
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.48
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.45
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 122, Estimated Rating: 3.44
- Places in category '122' with similar rating ---
- - No similar places found


**Top Recommendations for User 143**
- Recommended Category: 416, Estimated Rating: 4.26
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 431, Estimated Rating: 4.08
- Places in category '431' with similar rating ---
- Place: 431
- Place: Taman Hiburan Rakyat
- Description: Taman Hiburan Rakyat atau THR tentunya sudah tak asing lagi bagi masyarakat Surabaya. Taman ini berletak di belakang Taman Remaja Surabaya (TRS) dan juga ada di belakang Hi-Tech Mall. THR biasanya digunakan untuk pertunjukan kesenian daerah. Taman ini menjadi ikon Surabaya karena sejarahnya yang cukup menarik.
- City: Surabaya
- Price: 5000
- Rating: 4.20

- Recommended Category: 279, Estimated Rating: 4.01
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 230, Estimated Rating: 4.01
- Places in category '230' with similar rating ---
- Place: 230
- Place: Stone Garden Citatah
- Description: Stone  Garden, adalah sebutan nama untuk hamparan tanah yang diisi oleh  formasi batuan tak beraturan yang indah dan membentuk taman alam.Bandung  memiliki banyak wisata alam yang sangat luar biasa. Ternyata Bandung  memiliki wisata alam gua, yaitu Gua Pawon dan wisata Stone Garden. Gua  Pawon merupakan gua yang terbentuk pada zaman purba.Gua ini digunakan  oleh manusia purba sebagai tempat berlindung. Dan uniknya di atas Gua  Pawon terdapat hamparan bebatuan yang sangat indah dengan pemandangan  luar biasa dikenal dengan nama Stone Garden -Taman Batu. Tidak kalah  indah dengan wisata alam Tangkuban Perahu, Kawah Putih Ciwidey, dan Situ  Patengan merupakan kawasan wisata alam yang sering dikunjungi oleh  banyak wisatawan lokal maupun mancanegara. _x000D_
- City: Bandung
- Price: 30000
- Rating: 4.40

- Recommended Category: 94, Estimated Rating: 3.99
- Places in category '94' with similar rating ---
- - No similar places found


**Top Recommendations for User 144**
- Recommended Category: 416, Estimated Rating: 4.17
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 3.86
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 321, Estimated Rating: 3.86
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.79
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 230, Estimated Rating: 3.76
- Places in category '230' with similar rating ---
- - No similar places found


**Top Recommendations for User 145**
- Recommended Category: 279, Estimated Rating: 3.71
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 90, Estimated Rating: 3.66
- Places in category '90' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 3.66
- Places in category '401' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.63
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 230, Estimated Rating: 3.63
- Places in category '230' with similar rating ---
- - No similar places found


**Top Recommendations for User 146**
- Recommended Category: 416, Estimated Rating: 3.57
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.41
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.40
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 224, Estimated Rating: 3.34
- Places in category '224' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.33
- Places in category '1' with similar rating ---
- - No similar places found


**Top Recommendations for User 147**
- Recommended Category: 1, Estimated Rating: 3.85
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.73
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 178, Estimated Rating: 3.71
- Places in category '178' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.70
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 122, Estimated Rating: 3.69
- Places in category '122' with similar rating ---
- - No similar places found


**Top Recommendations for User 148**
- Recommended Category: 416, Estimated Rating: 4.09
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 422, Estimated Rating: 3.71
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.66
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.64
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.61
- Places in category '254' with similar rating ---
- - No similar places found


**Top Recommendations for User 149**
- Recommended Category: 416, Estimated Rating: 3.90
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.74
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 263, Estimated Rating: 3.67
- Places in category '263' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.67
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 122, Estimated Rating: 3.63
- Places in category '122' with similar rating ---
- - No similar places found


**Top Recommendations for User 150**
- Recommended Category: 1, Estimated Rating: 4.18
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 4.10
- Places in category '232' with similar rating ---
- Place: 232
- Place: Bukit Moko
- Description: Bandung sebagai destinasi wisata tak pernah ada habisnya. Didukung dengan lanskap yang cantik, kawasan Bandung mampu menarik perhatian wisatawan. Baik dari segi alam, budaya, kuliner, dan seni kreatif secara bersamaan. Dari sekian banyak tempat wisata yang tersedia, Bukit Moko Bandung menjadi salah satu yang cukup populer namanya dalam beberapa tahun belakangan. Berada di ketinggian sekitar 1500 mdpl, Bukit Moko memiliki cuaca yang sejuk. Bagi pengunjung yang tidak biasa di cuaca ini, ada baiknya membawa jaket tebal apalagi jika datang saat musim penghujan.
- City: Bandung
- Price: 25000
- Rating: 4.50

- Recommended Category: 416, Estimated Rating: 4.09
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 52, Estimated Rating: 4.04
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 253, Estimated Rating: 4.02
- Places in category '253' with similar rating ---
- - No similar places found


**Top Recommendations for User 151**
- Recommended Category: 1, Estimated Rating: 4.07
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 125, Estimated Rating: 4.03
- Places in category '125' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 4.00
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.99
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 255, Estimated Rating: 3.98
- Places in category '255' with similar rating ---
- - No similar places found


**Top Recommendations for User 152**
- Recommended Category: 416, Estimated Rating: 3.79
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.67
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.64
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.59
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.59
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 153**
- Recommended Category: 146, Estimated Rating: 3.59
- Places in category '146' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.50
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 138, Estimated Rating: 3.36
- Places in category '138' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.33
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.32
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 154**
- Recommended Category: 416, Estimated Rating: 4.24
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 122, Estimated Rating: 4.05
- Places in category '122' with similar rating ---
- Place: 122
- Place: Watu Goyang
- Description: Watu Goyang ini berasal dari Bahasa Jawa yang berarti Batu yang bergoyang. Dulunya di tempat ini terdapat batu yang berada di puncak yang bias bergoyang ketika disentuh maupun didorong. Batu tersebut sudah ada sejak ratuhan tahun yang lalu. Para pengelola menjelaskan bahwa Watu Goyang ini tidak pernah roboh walaupun disetuh bahkan di panjat, tetapi pihak pengelola melarang kesar untuk pengunjung yang mencoba menaiki Watu Goyang tersebut karena akan sangat berbahaya bagi keselamatan. Keunikan dari batu raksasa itu adalah stukturnya yang memuat tonjolan-tonjolan yang unik. Pemandangan alam yang masih hijau dan gunung merapi terlihat sangat alami, sangat cocok bagi para pengunjung untuk menghilangkan rasa bosan dengan aktivitas sehari-hari di perkotaan. Berada di puncak ini pengunjung akan merasakan sejuknya suhu udara, terutama pada pagi hari. Dan ketika di siang hari, meskipun cuaca sedang panas, maka hembusan angin yang menyapa terasa segar dan nyaman.
- City: Yogyakarta
- Price: 2500
- Rating: 4.40

- Recommended Category: 401, Estimated Rating: 3.98
- Places in category '401' with similar rating ---
- Place: 401
- Place: Taman Keputran
- Description: Ntah, mengapa nama taman ini disebut dengan taman keputran, namun jika dikaitkan dengan lokasi taman, berada di persimpangan jalan Keputran Surabaya. Selain itu, di seberang sungai terdapat juga sebuah taman yang bernama Taman Lalu Lintas Surabaya. Keunikan Taman Keputran dibandingkan dengan taman lainnya adalah adanya lampu yang kerangka tiangnya mirip dengan manusia, dengan gaya pose yang berbeda-beda itu, membuat taman ini terlihat ada sentuhan kreatif dan inovatif. Lampu-lampu itu terang dan indah pada malam hari. Untuknya, taman ini cukup representatif sebagai pilihan mendapatkan tempat tongkrongan yang tenang pada malam hari. Tak hanya itu, jalur pijat refleksi kaki juga tersedia disana. Cocok untuk pengunjung yang ingin mencoba pengobatan refleksi kaki sederhana. Melalui pemahaman tentang jalur refleksi secara mendalam, sebagian orang memilih sarana ini dalam usaha menyembuhkan penyakit._x000D_
- City: Surabaya
- Price: 0
- Rating: 4.30

- Recommended Category: 300, Estimated Rating: 3.96
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.93
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20


**Top Recommendations for User 155**
- Recommended Category: 254, Estimated Rating: 3.60
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 333, Estimated Rating: 3.60
- Places in category '333' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.58
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.51
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.51
- Places in category '44' with similar rating ---
- - No similar places found


**Top Recommendations for User 156**
- Recommended Category: 416, Estimated Rating: 4.17
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 134, Estimated Rating: 4.03
- Places in category '134' with similar rating ---
- Place: 134
- Place: Desa Wisata Gamplong
- Description: Desa Wisata Gamplong adalah desa wisata kerajinan tenun yang berada di Padukuhan Gamplong Desa Sumber Rahayu Kecamatan Moyudan Kabupaten Sleman, Yogyakarta. Desa wisata yang terletak di sebelah barat Kota Yogyakarta ini cukup menarik untuk disinggahi wisatawan terlebih karena masih adanya industri kerajinan tenun tradisional dengan menggunakan Alat Tenun Bukan Mesin (ATBM). Dengan ATBM, masyarakat perajin Gamplong mampu menghasilkan kain tenun sebagai bahan stagen (kain panjang untuk melilit bagian perut wanita). Selain kerajinan tenun, perajin juga mampu memproduksi kerajinan anyaman untuk suvenir.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 97, Estimated Rating: 4.01
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 4.00
- Places in category '321' with similar rating ---
- Place: 321
- Place: Glamping Lakeside Rancabali
- Description: Glamping Lakeside Rancabali menawarkan tempat berkemah mewah yang cocok untuk Anda jadikan destinasi liburan di akhir pekan. Seperti namanya, glamping atau glamour camping terdiri dari tenda yang berisikan tempat tidur nyaman, kamar mandi di dalam ruangan, pemanas, dan berbagai amenitas layaknya hotel berbintang. Berlokasi di Jalan Raya Ciwidey Rancabali KM 1, Glamping Lakeside berada di sekitar Situ Patenggang dan perkebunan teh yang subur. Tidak heran jika pemandangan dari glamping ini begitu apik dan Instagramable untuk berfoto.
- City: Bandung
- Price: 30000
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 3.99
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20


**Top Recommendations for User 157**
- Recommended Category: 139, Estimated Rating: 3.98
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.92
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.87
- Places in category '314' with similar rating ---
- Place: 314
- Place: Tafso Barn
- Description: Nama Punclut mungkin sudah cukup akrab di telinga wisatawan. Kawasan dataran tinggi di Bandung yang belakangan populer sebagai destinasi wisata. Selain menawarkan pemandangan yang indah, kawasan ini juga memiliki banyak tempat wisata untuk rekreasi. Salah satunya adalah kawasan wisata sekaligus restoran Tafso dan Boda Barn atau sering dikenal juga dengan Tafso Barn. Tempat rekreasi ini memadukan tempat makan dengan nuansa taman dan gudang yang unik. Pemandangan perbukitan yang dipotong garis horison menjadi incaran utama pengunjung. Sangat tepat nongkrong dan bersantai ketika berlibur di Bandung.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 230, Estimated Rating: 3.87
- Places in category '230' with similar rating ---
- - No similar places found

- Recommended Category: 138, Estimated Rating: 3.85
- Places in category '138' with similar rating ---
- - No similar places found


**Top Recommendations for User 158**
- Recommended Category: 416, Estimated Rating: 3.89
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.77
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 3.74
- Places in category '401' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.70
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.63
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 159**
- Recommended Category: 98, Estimated Rating: 3.55
- Places in category '98' with similar rating ---
- - No similar places found

- Recommended Category: 94, Estimated Rating: 3.54
- Places in category '94' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.52
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.50
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.48
- Places in category '97' with similar rating ---
- - No similar places found


**Top Recommendations for User 160**
- Recommended Category: 416, Estimated Rating: 3.61
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 3.61
- Places in category '83' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.57
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.53
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 353, Estimated Rating: 3.50
- Places in category '353' with similar rating ---
- - No similar places found


**Top Recommendations for User 161**
- Recommended Category: 97, Estimated Rating: 3.84
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.78
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 3.73
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.71
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 92, Estimated Rating: 3.64
- Places in category '92' with similar rating ---
- - No similar places found


**Top Recommendations for User 162**
- Recommended Category: 300, Estimated Rating: 3.96
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 90, Estimated Rating: 3.95
- Places in category '90' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.92
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.91
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.86
- Places in category '97' with similar rating ---
- - No similar places found


**Top Recommendations for User 163**
- Recommended Category: 416, Estimated Rating: 3.91
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 224, Estimated Rating: 3.80
- Places in category '224' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.80
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.75
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.72
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 164**
- Recommended Category: 139, Estimated Rating: 4.04
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.74
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 178, Estimated Rating: 3.73
- Places in category '178' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.73
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 253, Estimated Rating: 3.71
- Places in category '253' with similar rating ---
- - No similar places found


**Top Recommendations for User 165**
- Recommended Category: 254, Estimated Rating: 3.91
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 139, Estimated Rating: 3.88
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 172, Estimated Rating: 3.87
- Places in category '172' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.87
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.86
- Places in category '28' with similar rating ---
- - No similar places found


**Top Recommendations for User 166**
- Recommended Category: 254, Estimated Rating: 4.01
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 416, Estimated Rating: 3.93
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.87
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.87
- Places in category '132' with similar rating ---
- - No similar places found

- Recommended Category: 253, Estimated Rating: 3.74
- Places in category '253' with similar rating ---
- - No similar places found


**Top Recommendations for User 167**
- Recommended Category: 416, Estimated Rating: 4.02
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 139, Estimated Rating: 3.88
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.88
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.85
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.84
- Places in category '254' with similar rating ---
- - No similar places found


**Top Recommendations for User 168**
- Recommended Category: 52, Estimated Rating: 4.16
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 224, Estimated Rating: 4.12
- Places in category '224' with similar rating ---
- Place: 224
- Place: Dago Dreampark
- Description: Dago Dreampark merupakan wisata kekinian di Kota Bandung dengan luas 13 hektar yang mengusung konsep Jawa - Sunda & Bali dengan dilengkapi berbagai fasilitas & wahana yang menarik.
- City: Bandung
- Price: 40000
- Rating: 4.20

- Recommended Category: 401, Estimated Rating: 4.08
- Places in category '401' with similar rating ---
- Place: 401
- Place: Taman Keputran
- Description: Ntah, mengapa nama taman ini disebut dengan taman keputran, namun jika dikaitkan dengan lokasi taman, berada di persimpangan jalan Keputran Surabaya. Selain itu, di seberang sungai terdapat juga sebuah taman yang bernama Taman Lalu Lintas Surabaya. Keunikan Taman Keputran dibandingkan dengan taman lainnya adalah adanya lampu yang kerangka tiangnya mirip dengan manusia, dengan gaya pose yang berbeda-beda itu, membuat taman ini terlihat ada sentuhan kreatif dan inovatif. Lampu-lampu itu terang dan indah pada malam hari. Untuknya, taman ini cukup representatif sebagai pilihan mendapatkan tempat tongkrongan yang tenang pada malam hari. Tak hanya itu, jalur pijat refleksi kaki juga tersedia disana. Cocok untuk pengunjung yang ingin mencoba pengobatan refleksi kaki sederhana. Melalui pemahaman tentang jalur refleksi secara mendalam, sebagian orang memilih sarana ini dalam usaha menyembuhkan penyakit._x000D_
- City: Surabaya
- Price: 0
- Rating: 4.30

- Recommended Category: 322, Estimated Rating: 4.06
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 115, Estimated Rating: 4.04
- Places in category '115' with similar rating ---
- Place: 115
- Place: Monumen Sanapati
- Description: Monumen Sanapati dibangun untuk memeringati 50 tahun persandian Indonesia di Yogyakarta. Persandian di Yogyakarta memiliki andil yang besar dalam mempertahankan kemerdekaan Negara Indonesia. Tercatat pada peristiwa Serangan Umum 1 Maret 1949, sandi buatan Roebiono masih digunakan untuk menyebarkan pesan pada dunia bahwa Indonesia masih ada dan masih bisa melawan. Selanjutnya, peristiwa tersebut dikenal dengan Peristiwa 6 jam di Yogyakarta. Monumen Sanapati terletak di kawasan Kotabaru, tepatnya di tengah persimpangan Jalan Abu Bakar Ali, di taman depan Gereja Santo Antonius. Bangunan Monumen Sanapati berbentuk segitiga berujung lancip menyerupai piramida dengan tinggi 3 meter dan lebar 2,5 meter. Monumen tersebut didesain oleh Drs Kasman dan diresmikan oleh Menteri Sekretaris Negara Republik Indonesia saat itu, Moerdiono, pada 4 April 1996, tepat pada Hari Persandian Indonesia. Pembangunan monument membutuhkan waktu satu bulan dan memakan biaya sebesar 25 juta rupiah.
- City: Yogyakarta
- Price: 15000
- Rating: 4.30


**Top Recommendations for User 169**
- Recommended Category: 52, Estimated Rating: 4.03
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.78
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.77
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.77
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 51, Estimated Rating: 3.76
- Places in category '51' with similar rating ---
- Place: 51
- Place: Jakarta Planetarium
- Description: Planetarium dan Observatorium Jakarta adalah satu dari tiga wahana simulasi langit di Indonesia selain di Kutai, Kalimantan Timur, dan Surabaya, Jawa Timur. Planetarium tertua ini letaknya di Taman Ismail Marzuki, Jakarta. Planetarium Jakarta merupakan sarana wisata pendidikan yang dapat menyajikan pertunjukan / peragaan simulasi perbintangan atau benda-benda langit. Pengunjung diajak mengembara di jagat raya untuk memahami konsepsi tentang alam semesta melalui acara demi acara. Planetarium Jakarta berdiri tahun 1964 diprakarsai Presiden Soekarno dan diserahkan ke Pemerintah Provinsi DKI Jakarta pada 1969. Di tempat ini juga tersedia ruang pameran benda- benda angkasa yang menyuguhkan berbagai foto serta keterangan lengkap dari berbagai bentuk galaksi, teori-teori pembentukan galaksi disertai pengenalan tokoh-tokoh di balik munculnya teori.
- City: Jakarta
- Price: 12000
- Rating: 4.10


**Top Recommendations for User 170**
- Recommended Category: 416, Estimated Rating: 3.62
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.54
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.45
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.44
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 117, Estimated Rating: 3.40
- Places in category '117' with similar rating ---
- - No similar places found


**Top Recommendations for User 171**
- Recommended Category: 416, Estimated Rating: 3.98
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.68
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.61
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.56
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 282, Estimated Rating: 3.56
- Places in category '282' with similar rating ---
- - No similar places found


**Top Recommendations for User 172**
- Recommended Category: 416, Estimated Rating: 3.84
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.79
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.75
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.75
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 167, Estimated Rating: 3.70
- Places in category '167' with similar rating ---
- - No similar places found


**Top Recommendations for User 173**
- Recommended Category: 138, Estimated Rating: 3.44
- Places in category '138' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 3.43
- Places in category '79' with similar rating ---
- - No similar places found

- Recommended Category: 208, Estimated Rating: 3.40
- Places in category '208' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.40
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.37
- Places in category '97' with similar rating ---
- - No similar places found


**Top Recommendations for User 174**
- Recommended Category: 157, Estimated Rating: 3.80
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.59
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.54
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.54
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 80, Estimated Rating: 3.50
- Places in category '80' with similar rating ---
- - No similar places found


**Top Recommendations for User 175**
- Recommended Category: 416, Estimated Rating: 3.99
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.92
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 157, Estimated Rating: 3.88
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.87
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 164, Estimated Rating: 3.87
- Places in category '164' with similar rating ---
- - No similar places found


**Top Recommendations for User 176**
- Recommended Category: 416, Estimated Rating: 4.11
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 157, Estimated Rating: 3.86
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.76
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 226, Estimated Rating: 3.75
- Places in category '226' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.74
- Places in category '112' with similar rating ---
- - No similar places found


**Top Recommendations for User 177**
- Recommended Category: 1, Estimated Rating: 4.28
- Places in category '1' with similar rating ---
- Place: 1
- Place: Monumen Nasional
- Description: Monumen Nasional atau yang populer disingkat dengan Monas atau Tugu Monas adalah monumen peringatan setinggi 132 meter (433 kaki) yang didirikan untuk mengenang perlawanan dan perjuangan rakyat Indonesia untuk merebut kemerdekaan dari pemerintahan kolonial Hindia Belanda. Pembangunan monumen ini dimulai pada tanggal 17 Agustus 1961 di bawah perintah presiden Soekarno dan dibuka untuk umum pada tanggal 12 Juli 1975. Tugu ini dimahkotai lidah api yang dilapisi lembaran emas yang melambangkan semangat perjuangan yang menyala-nyala. Monumen Nasional terletak tepat di tengah Lapangan Medan Merdeka, Jakarta Pusat.
- City: Jakarta
- Price: 20000
- Rating: 4.60

- Recommended Category: 416, Estimated Rating: 4.19
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 97, Estimated Rating: 4.06
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 4.01
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 4.00
- Places in category '321' with similar rating ---
- - No similar places found


**Top Recommendations for User 178**
- Recommended Category: 322, Estimated Rating: 3.78
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 136, Estimated Rating: 3.76
- Places in category '136' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.73
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.70
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.68
- Places in category '300' with similar rating ---
- - No similar places found


**Top Recommendations for User 179**
- Recommended Category: 416, Estimated Rating: 4.13
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 80, Estimated Rating: 3.91
- Places in category '80' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.86
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 8, Estimated Rating: 3.85
- Places in category '8' with similar rating ---
- Place: 8
- Place: Ocean Ecopark
- Description: Ocean Ecopark Salah satu zona rekreasi Ancol yang menawarkan ruang terbuka hijau serta pengalaman dan tidak melupakan sisi pendidikan bagi para pengunjung. Beragam aktivitas dan wahana seru yang bisa dimainkan di tempat rekreasi keluarga, beberapa diantaranya : Outbondholic, Rumah Energy, Rumah Lebah, Kano, Paintball, Outbondholic, Eco-Market, Learning Farm, Eco-Bike, Faunaland. Kalau berencana berwisata ke Ocean Eco Park ancol tidak perlu khawatir dengan harga tiket masuknya. Karena harga tiket masuk Ocean Eco Park ditawarkan dengan harga yang masih terjangkau._x000D_
- City: Jakarta
- Price: 180000
- Rating: 4.00

- Recommended Category: 387, Estimated Rating: 3.84
- Places in category '387' with similar rating ---
- - No similar places found


**Top Recommendations for User 180**
- Recommended Category: 416, Estimated Rating: 3.85
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 146, Estimated Rating: 3.51
- Places in category '146' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.45
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.42
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 263, Estimated Rating: 3.37
- Places in category '263' with similar rating ---
- - No similar places found


**Top Recommendations for User 181**
- Recommended Category: 416, Estimated Rating: 3.90
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.72
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 94, Estimated Rating: 3.69
- Places in category '94' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.69
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.67
- Places in category '254' with similar rating ---
- - No similar places found


**Top Recommendations for User 182**
- Recommended Category: 157, Estimated Rating: 3.84
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.69
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.62
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.60
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.59
- Places in category '132' with similar rating ---
- - No similar places found


**Top Recommendations for User 183**
- Recommended Category: 322, Estimated Rating: 3.52
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.50
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.38
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 3.36
- Places in category '79' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.35
- Places in category '332' with similar rating ---
- - No similar places found


**Top Recommendations for User 184**
- Recommended Category: 138, Estimated Rating: 3.84
- Places in category '138' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.77
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.74
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 210, Estimated Rating: 3.67
- Places in category '210' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.61
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 185**
- Recommended Category: 97, Estimated Rating: 3.83
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.81
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.75
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 91, Estimated Rating: 3.67
- Places in category '91' with similar rating ---
- - No similar places found

- Recommended Category: 295, Estimated Rating: 3.64
- Places in category '295' with similar rating ---
- - No similar places found


**Top Recommendations for User 186**
- Recommended Category: 112, Estimated Rating: 4.09
- Places in category '112' with similar rating ---
- Place: 112
- Place: Bukit Bintang Yogyakarta
- Description: Bukit Bintang merupakan salah satu lokasi nongkrong favorit di Yogyakarta. Saat malam tiba, pemandangan Yogyakarta sangatlah indah!Terletak di perbatasan Bantul dan Gunungkidul, siapa pun yang berkunjung ke kawasan ini dapat menikmati taburan gemintang di langit malam serta kerlip benderang lampu kota dari ketinggian.Kota ini memiliki Bukit Bintang yang selalu ramai dipadati kawula muda untuk menikmati keindahan malam. Bukit Bintang menjadi tempat yang sempurna untuk menikmati senja hingga malam tiba._x000D_
- City: Yogyakarta
- Price: 25000
- Rating: 4.50

- Recommended Category: 94, Estimated Rating: 4.02
- Places in category '94' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.99
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 230, Estimated Rating: 3.93
- Places in category '230' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.90
- Places in category '314' with similar rating ---
- Place: 314
- Place: Tafso Barn
- Description: Nama Punclut mungkin sudah cukup akrab di telinga wisatawan. Kawasan dataran tinggi di Bandung yang belakangan populer sebagai destinasi wisata. Selain menawarkan pemandangan yang indah, kawasan ini juga memiliki banyak tempat wisata untuk rekreasi. Salah satunya adalah kawasan wisata sekaligus restoran Tafso dan Boda Barn atau sering dikenal juga dengan Tafso Barn. Tempat rekreasi ini memadukan tempat makan dengan nuansa taman dan gudang yang unik. Pemandangan perbukitan yang dipotong garis horison menjadi incaran utama pengunjung. Sangat tepat nongkrong dan bersantai ketika berlibur di Bandung.
- City: Bandung
- Price: 0
- Rating: 4.20


**Top Recommendations for User 187**
- Recommended Category: 52, Estimated Rating: 3.78
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.77
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 431, Estimated Rating: 3.69
- Places in category '431' with similar rating ---
- - No similar places found

- Recommended Category: 164, Estimated Rating: 3.68
- Places in category '164' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.66
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 188**
- Recommended Category: 224, Estimated Rating: 3.28
- Places in category '224' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.23
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.21
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.18
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.15
- Places in category '321' with similar rating ---
- - No similar places found


**Top Recommendations for User 189**
- Recommended Category: 139, Estimated Rating: 4.32
- Places in category '139' with similar rating ---
- Place: 139
- Place: Puncak Gunung Api Purba - Nglanggeran
- Description: Gunung Nglanggeran adalah sebuah gunung di Daerah Istimewa Yogyakarta, Indonesia. Gunung ini merupakan suatu gunung api purba yang terbentuk sekitar 0,6-70 juta tahun yang lalu atau yang memiliki umur tersier (Oligo-Miosen). Gunung Nglanggeran memiliki batuan yang sangat khas karena didominasi oleh aglomerat dan breksi gunung api. Gunung ini terletak di Desa Nglanggeran, Kecamatan Patuk, Kabupaten Gunung Kidul yang berada pada deretan Pegunungan Baturagung.
- City: Yogyakarta
- Price: 10000
- Rating: 4.70

- Recommended Category: 52, Estimated Rating: 4.22
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 97, Estimated Rating: 4.16
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 422, Estimated Rating: 4.14
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 4.08
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30


**Top Recommendations for User 190**
- Recommended Category: 416, Estimated Rating: 3.95
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.81
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.78
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.74
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 253, Estimated Rating: 3.72
- Places in category '253' with similar rating ---
- - No similar places found


**Top Recommendations for User 191**
- Recommended Category: 416, Estimated Rating: 3.83
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.76
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.70
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.67
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.66
- Places in category '132' with similar rating ---
- - No similar places found


**Top Recommendations for User 192**
- Recommended Category: 416, Estimated Rating: 3.87
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.72
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.72
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.70
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.69
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 193**
- Recommended Category: 97, Estimated Rating: 3.78
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.70
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.65
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.61
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.61
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 194**
- Recommended Category: 134, Estimated Rating: 3.92
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 387, Estimated Rating: 3.74
- Places in category '387' with similar rating ---
- - No similar places found

- Recommended Category: 333, Estimated Rating: 3.73
- Places in category '333' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.69
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 3.66
- Places in category '83' with similar rating ---
- - No similar places found


**Top Recommendations for User 195**
- Recommended Category: 112, Estimated Rating: 3.71
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 431, Estimated Rating: 3.67
- Places in category '431' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.67
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.66
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 387, Estimated Rating: 3.65
- Places in category '387' with similar rating ---
- - No similar places found


**Top Recommendations for User 196**
- Recommended Category: 322, Estimated Rating: 3.73
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.70
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.66
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.65
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.62
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 197**
- Recommended Category: 134, Estimated Rating: 3.98
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.97
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 399, Estimated Rating: 3.96
- Places in category '399' with similar rating ---
- - No similar places found

- Recommended Category: 164, Estimated Rating: 3.95
- Places in category '164' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.89
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 198**
- Recommended Category: 401, Estimated Rating: 3.71
- Places in category '401' with similar rating ---
- - No similar places found

- Recommended Category: 224, Estimated Rating: 3.67
- Places in category '224' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.64
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.58
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.54
- Places in category '321' with similar rating ---
- - No similar places found


**Top Recommendations for User 199**
- Recommended Category: 52, Estimated Rating: 3.83
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.69
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.66
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.66
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.64
- Places in category '322' with similar rating ---
- - No similar places found


**Top Recommendations for User 200**
- Recommended Category: 83, Estimated Rating: 3.78
- Places in category '83' with similar rating ---
- - No similar places found

- Recommended Category: 146, Estimated Rating: 3.65
- Places in category '146' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.59
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.55
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.53
- Places in category '314' with similar rating ---
- - No similar places found


**Top Recommendations for User 201**
- Recommended Category: 416, Estimated Rating: 4.10
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 139, Estimated Rating: 3.93
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.87
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 420, Estimated Rating: 3.87
- Places in category '420' with similar rating ---
- - No similar places found

- Recommended Category: 387, Estimated Rating: 3.85
- Places in category '387' with similar rating ---
- - No similar places found


**Top Recommendations for User 202**
- Recommended Category: 52, Estimated Rating: 4.11
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 157, Estimated Rating: 4.09
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 416, Estimated Rating: 4.01
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 183, Estimated Rating: 4.00
- Places in category '183' with similar rating ---
- - No similar places found

- Recommended Category: 115, Estimated Rating: 3.99
- Places in category '115' with similar rating ---
- Place: 115
- Place: Monumen Sanapati
- Description: Monumen Sanapati dibangun untuk memeringati 50 tahun persandian Indonesia di Yogyakarta. Persandian di Yogyakarta memiliki andil yang besar dalam mempertahankan kemerdekaan Negara Indonesia. Tercatat pada peristiwa Serangan Umum 1 Maret 1949, sandi buatan Roebiono masih digunakan untuk menyebarkan pesan pada dunia bahwa Indonesia masih ada dan masih bisa melawan. Selanjutnya, peristiwa tersebut dikenal dengan Peristiwa 6 jam di Yogyakarta. Monumen Sanapati terletak di kawasan Kotabaru, tepatnya di tengah persimpangan Jalan Abu Bakar Ali, di taman depan Gereja Santo Antonius. Bangunan Monumen Sanapati berbentuk segitiga berujung lancip menyerupai piramida dengan tinggi 3 meter dan lebar 2,5 meter. Monumen tersebut didesain oleh Drs Kasman dan diresmikan oleh Menteri Sekretaris Negara Republik Indonesia saat itu, Moerdiono, pada 4 April 1996, tepat pada Hari Persandian Indonesia. Pembangunan monument membutuhkan waktu satu bulan dan memakan biaya sebesar 25 juta rupiah.
- City: Yogyakarta
- Price: 15000
- Rating: 4.30


**Top Recommendations for User 203**
- Recommended Category: 416, Estimated Rating: 4.08
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 253, Estimated Rating: 3.81
- Places in category '253' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.72
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.69
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.64
- Places in category '322' with similar rating ---
- - No similar places found


**Top Recommendations for User 204**
- Recommended Category: 416, Estimated Rating: 3.81
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 312, Estimated Rating: 3.76
- Places in category '312' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.73
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 160, Estimated Rating: 3.69
- Places in category '160' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 3.66
- Places in category '422' with similar rating ---
- - No similar places found


**Top Recommendations for User 205**
- Recommended Category: 416, Estimated Rating: 4.16
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 1, Estimated Rating: 3.96
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.89
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.87
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.84
- Places in category '52' with similar rating ---
- - No similar places found


**Top Recommendations for User 206**
- Recommended Category: 97, Estimated Rating: 3.91
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.85
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 3.80
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.71
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 3.69
- Places in category '79' with similar rating ---
- - No similar places found


**Top Recommendations for User 207**
- Recommended Category: 139, Estimated Rating: 3.81
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.79
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.73
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.69
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.64
- Places in category '52' with similar rating ---
- - No similar places found


**Top Recommendations for User 208**
- Recommended Category: 139, Estimated Rating: 4.19
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 4.16
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 279, Estimated Rating: 4.11
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 4.05
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 4.03
- Places in category '232' with similar rating ---
- - No similar places found


**Top Recommendations for User 209**
- Recommended Category: 416, Estimated Rating: 4.11
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 52, Estimated Rating: 3.95
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 160, Estimated Rating: 3.86
- Places in category '160' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 3.80
- Places in category '83' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.80
- Places in category '132' with similar rating ---
- - No similar places found


**Top Recommendations for User 210**
- Recommended Category: 422, Estimated Rating: 3.64
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.63
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 224, Estimated Rating: 3.55
- Places in category '224' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.50
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 90, Estimated Rating: 3.48
- Places in category '90' with similar rating ---
- - No similar places found


**Top Recommendations for User 211**
- Recommended Category: 279, Estimated Rating: 4.09
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.93
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.81
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.76
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.73
- Places in category '300' with similar rating ---
- - No similar places found


**Top Recommendations for User 212**
- Recommended Category: 139, Estimated Rating: 4.09
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.97
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.96
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.92
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.89
- Places in category '332' with similar rating ---
- - No similar places found


**Top Recommendations for User 213**
- Recommended Category: 416, Estimated Rating: 3.98
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.89
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.78
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 3.77
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.77
- Places in category '52' with similar rating ---
- - No similar places found


**Top Recommendations for User 214**
- Recommended Category: 157, Estimated Rating: 3.90
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.88
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 117, Estimated Rating: 3.87
- Places in category '117' with similar rating ---
- Place: 117
- Place: The World Landmarks - Merapi Park Yogyakarta
- Description: Merapi Park merupakan salah satu tempat wisata di Yogyakarta yang terletak di Jalan Kaliurang km 22, Hargobinangun, Kecamatan Pakem, Kabupaten Sleman. Pengoperasian Merapi Park dimulai sejak tanggal 25 Juni 2017. Fasilitas yang tersedia meliputi tempat pengambilan foto
- City: Yogyakarta
- Price: 22000
- Rating: 4.20

- Recommended Category: 300, Estimated Rating: 3.81
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.75
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 215**
- Recommended Category: 416, Estimated Rating: 3.93
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.77
- Places in category '132' with similar rating ---
- - No similar places found

- Recommended Category: 142, Estimated Rating: 3.76
- Places in category '142' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.73
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 308, Estimated Rating: 3.70
- Places in category '308' with similar rating ---
- - No similar places found


**Top Recommendations for User 216**
- Recommended Category: 157, Estimated Rating: 4.17
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 416, Estimated Rating: 4.00
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.93
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 387, Estimated Rating: 3.88
- Places in category '387' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.88
- Places in category '300' with similar rating ---
- - No similar places found


**Top Recommendations for User 217**
- Recommended Category: 416, Estimated Rating: 3.57
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.45
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.44
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 243, Estimated Rating: 3.42
- Places in category '243' with similar rating ---
- - No similar places found

- Recommended Category: 53, Estimated Rating: 3.41
- Places in category '53' with similar rating ---
- - No similar places found


**Top Recommendations for User 218**
- Recommended Category: 112, Estimated Rating: 3.48
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.44
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.34
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.32
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.28
- Places in category '416' with similar rating ---
- - No similar places found


**Top Recommendations for User 219**
- Recommended Category: 52, Estimated Rating: 3.78
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.73
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.67
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.65
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 333, Estimated Rating: 3.62
- Places in category '333' with similar rating ---
- - No similar places found


**Top Recommendations for User 220**
- Recommended Category: 254, Estimated Rating: 3.39
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 251, Estimated Rating: 3.36
- Places in category '251' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.35
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.27
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.25
- Places in category '322' with similar rating ---
- - No similar places found


**Top Recommendations for User 221**
- Recommended Category: 139, Estimated Rating: 3.82
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.75
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.52
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.51
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.50
- Places in category '416' with similar rating ---
- - No similar places found


**Top Recommendations for User 222**
- Recommended Category: 117, Estimated Rating: 3.77
- Places in category '117' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.69
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.67
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.67
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 53, Estimated Rating: 3.65
- Places in category '53' with similar rating ---
- - No similar places found


**Top Recommendations for User 223**
- Recommended Category: 254, Estimated Rating: 3.99
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 416, Estimated Rating: 3.97
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 399, Estimated Rating: 3.85
- Places in category '399' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.82
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 333, Estimated Rating: 3.78
- Places in category '333' with similar rating ---
- - No similar places found


**Top Recommendations for User 224**
- Recommended Category: 1, Estimated Rating: 4.04
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.98
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 94, Estimated Rating: 3.86
- Places in category '94' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.83
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 3.83
- Places in category '401' with similar rating ---
- - No similar places found


**Top Recommendations for User 225**
- Recommended Category: 139, Estimated Rating: 4.07
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.97
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.92
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 53, Estimated Rating: 3.92
- Places in category '53' with similar rating ---
- - No similar places found

- Recommended Category: 115, Estimated Rating: 3.91
- Places in category '115' with similar rating ---
- - No similar places found


**Top Recommendations for User 226**
- Recommended Category: 416, Estimated Rating: 4.15
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 157, Estimated Rating: 4.05
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.00
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 132, Estimated Rating: 3.95
- Places in category '132' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.91
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 227**
- Recommended Category: 416, Estimated Rating: 3.99
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 60, Estimated Rating: 3.90
- Places in category '60' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.90
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 3.84
- Places in category '79' with similar rating ---
- - No similar places found

- Recommended Category: 178, Estimated Rating: 3.83
- Places in category '178' with similar rating ---
- - No similar places found


**Top Recommendations for User 228**
- Recommended Category: 416, Estimated Rating: 4.09
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 94, Estimated Rating: 3.95
- Places in category '94' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.91
- Places in category '132' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.89
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.89
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 229**
- Recommended Category: 254, Estimated Rating: 3.72
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 164, Estimated Rating: 3.58
- Places in category '164' with similar rating ---
- - No similar places found

- Recommended Category: 91, Estimated Rating: 3.51
- Places in category '91' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.51
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 350, Estimated Rating: 3.50
- Places in category '350' with similar rating ---
- - No similar places found


**Top Recommendations for User 230**
- Recommended Category: 322, Estimated Rating: 4.00
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 232, Estimated Rating: 3.97
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.82
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.75
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 36, Estimated Rating: 3.72
- Places in category '36' with similar rating ---
- - No similar places found


**Top Recommendations for User 231**
- Recommended Category: 139, Estimated Rating: 3.71
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.71
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 117, Estimated Rating: 3.69
- Places in category '117' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.63
- Places in category '314' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.58
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 232**
- Recommended Category: 416, Estimated Rating: 3.77
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.60
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.60
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.60
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.57
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 233**
- Recommended Category: 416, Estimated Rating: 3.98
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.93
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 422, Estimated Rating: 3.87
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.83
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.80
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 234**
- Recommended Category: 322, Estimated Rating: 3.76
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.66
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 160, Estimated Rating: 3.47
- Places in category '160' with similar rating ---
- - No similar places found

- Recommended Category: 60, Estimated Rating: 3.44
- Places in category '60' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.42
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 235**
- Recommended Category: 279, Estimated Rating: 3.97
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.95
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 94, Estimated Rating: 3.89
- Places in category '94' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.86
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.85
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 236**
- Recommended Category: 52, Estimated Rating: 3.76
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.75
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 407, Estimated Rating: 3.66
- Places in category '407' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.60
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.56
- Places in category '321' with similar rating ---
- - No similar places found


**Top Recommendations for User 237**
- Recommended Category: 157, Estimated Rating: 3.69
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.65
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.60
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.54
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.52
- Places in category '322' with similar rating ---
- - No similar places found


**Top Recommendations for User 238**
- Recommended Category: 112, Estimated Rating: 4.22
- Places in category '112' with similar rating ---
- Place: 112
- Place: Bukit Bintang Yogyakarta
- Description: Bukit Bintang merupakan salah satu lokasi nongkrong favorit di Yogyakarta. Saat malam tiba, pemandangan Yogyakarta sangatlah indah!Terletak di perbatasan Bantul dan Gunungkidul, siapa pun yang berkunjung ke kawasan ini dapat menikmati taburan gemintang di langit malam serta kerlip benderang lampu kota dari ketinggian.Kota ini memiliki Bukit Bintang yang selalu ramai dipadati kawula muda untuk menikmati keindahan malam. Bukit Bintang menjadi tempat yang sempurna untuk menikmati senja hingga malam tiba._x000D_
- City: Yogyakarta
- Price: 25000
- Rating: 4.50

- Recommended Category: 44, Estimated Rating: 3.97
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.93
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 263, Estimated Rating: 3.90
- Places in category '263' with similar rating ---
- Place: 263
- Place: Curug Batu Templek
- Description: Curug Batu Templek Bandung adalah sebuah wisata alam air terjun yang terletak di Kota Bandung Timur. Dari sekian banyak wisata alam air terjun di Bandung, Curug Batu Templek tak kalah menarik karena pemandangan di sekitarnya dipenuhi dengan terbing yang berbatu cadas. Sejarah Curug Batu Templek sangat sederhana, hanya karena di tempat ini dahulunya terdapat penambangan batu yang terdapat sebuah aliran air terjun dan diketahui aliran tersebut berasal dari sungai yang ada di atas tebing sehingga dinamakan Curug Batu Templek.
- City: Bandung
- Price: 5000
- Rating: 4.10

- Recommended Category: 97, Estimated Rating: 3.87
- Places in category '97' with similar rating ---
- - No similar places found


**Top Recommendations for User 239**
- Recommended Category: 52, Estimated Rating: 4.13
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 416, Estimated Rating: 4.12
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.09
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 401, Estimated Rating: 3.92
- Places in category '401' with similar rating ---
- Place: 401
- Place: Taman Keputran
- Description: Ntah, mengapa nama taman ini disebut dengan taman keputran, namun jika dikaitkan dengan lokasi taman, berada di persimpangan jalan Keputran Surabaya. Selain itu, di seberang sungai terdapat juga sebuah taman yang bernama Taman Lalu Lintas Surabaya. Keunikan Taman Keputran dibandingkan dengan taman lainnya adalah adanya lampu yang kerangka tiangnya mirip dengan manusia, dengan gaya pose yang berbeda-beda itu, membuat taman ini terlihat ada sentuhan kreatif dan inovatif. Lampu-lampu itu terang dan indah pada malam hari. Untuknya, taman ini cukup representatif sebagai pilihan mendapatkan tempat tongkrongan yang tenang pada malam hari. Tak hanya itu, jalur pijat refleksi kaki juga tersedia disana. Cocok untuk pengunjung yang ingin mencoba pengobatan refleksi kaki sederhana. Melalui pemahaman tentang jalur refleksi secara mendalam, sebagian orang memilih sarana ini dalam usaha menyembuhkan penyakit._x000D_
- City: Surabaya
- Price: 0
- Rating: 4.30

- Recommended Category: 146, Estimated Rating: 3.88
- Places in category '146' with similar rating ---
- - No similar places found


**Top Recommendations for User 240**
- Recommended Category: 134, Estimated Rating: 3.75
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.74
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.68
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 287, Estimated Rating: 3.65
- Places in category '287' with similar rating ---
- - No similar places found

- Recommended Category: 91, Estimated Rating: 3.64
- Places in category '91' with similar rating ---
- - No similar places found


**Top Recommendations for User 241**
- Recommended Category: 300, Estimated Rating: 3.93
- Places in category '300' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.91
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.82
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 117, Estimated Rating: 3.79
- Places in category '117' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.79
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 242**
- Recommended Category: 254, Estimated Rating: 4.15
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 279, Estimated Rating: 4.15
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 4.12
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 4.10
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 232, Estimated Rating: 4.08
- Places in category '232' with similar rating ---
- - No similar places found


**Top Recommendations for User 243**
- Recommended Category: 83, Estimated Rating: 3.87
- Places in category '83' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.79
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 251, Estimated Rating: 3.78
- Places in category '251' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.75
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 53, Estimated Rating: 3.74
- Places in category '53' with similar rating ---
- - No similar places found


**Top Recommendations for User 244**
- Recommended Category: 157, Estimated Rating: 3.88
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.85
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 91, Estimated Rating: 3.80
- Places in category '91' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.79
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.73
- Places in category '321' with similar rating ---
- - No similar places found


**Top Recommendations for User 245**
- Recommended Category: 416, Estimated Rating: 3.76
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.70
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.63
- Places in category '314' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.63
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.63
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 246**
- Recommended Category: 139, Estimated Rating: 3.98
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 249, Estimated Rating: 3.94
- Places in category '249' with similar rating ---
- Place: 249
- Place: Upside Down World Bandung
- Description: Upside Down World Bandung pertama kali dibuka pada 10 Oktober 2016 dengan konsep yang sangat unik. Semua furnitur yang berada di Upside Down World ini dipasang terbalik seolah pengunjung berada di langit-langit ruangan. Dengan konsep rumah terbaik tempat wisata ini memiliki keunikan tersendiri. Pengunjung yang datang ke sini kebanyakan para kawula muda yang ingin berfoto dengan latar yang tak biasa. Di dalam bangunan ini terdapat sepuluh ruangan dengan tema yang berbeda-beda layaknya di dalam sebuah rumah. Ada ruang tamu, kamar tidur anak, ruang keluarga, ruang santai, dapur, kamar mandi, dan lain sebagainya. Pengunjung bisa mengambil gambar pada semua ruangan dan hasil foto dapat dicetak di layanan cetak foto. Jika pengunjung merasa tak biasa dan bingung harus bergaya foto seperti apa, maka akan ada petugas yang dapat mengarahkan gayanya. Atau dengan melihat hasil foto para pengunjung sebelumnya yang dipasang di ruangan itu._x000D_
- City: Bandung
- Price: 100000
- Rating: 4.00

- Recommended Category: 1, Estimated Rating: 3.90
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.81
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.79
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 247**
- Recommended Category: 399, Estimated Rating: 4.18
- Places in category '399' with similar rating ---
- Place: 399
- Place: Taman Pelangi
- Description: Kalau pelangi biasanya ada di siang hari pasca hujan, maka di Taman Pelangi Yogyakarta pengunjung justru bisa menikmatinya setiap malam hari. Ya, ini lantaran taman ini merupakan Taman Lampion beraneka warna dan rupa. Buka sejak petang, Taman ini memang sangat cocok dijadikan pilihan destinasi rekreasi keluarga saat malam hari. Namun karena merupakan taman outdoor alias terbuka, maka pilihan paling tepat untuk datang ialah saat cuaca tidak hujan.
- City: Surabaya
- Price: 0
- Rating: 4.50

- Recommended Category: 263, Estimated Rating: 3.96
- Places in category '263' with similar rating ---
- Place: 263
- Place: Curug Batu Templek
- Description: Curug Batu Templek Bandung adalah sebuah wisata alam air terjun yang terletak di Kota Bandung Timur. Dari sekian banyak wisata alam air terjun di Bandung, Curug Batu Templek tak kalah menarik karena pemandangan di sekitarnya dipenuhi dengan terbing yang berbatu cadas. Sejarah Curug Batu Templek sangat sederhana, hanya karena di tempat ini dahulunya terdapat penambangan batu yang terdapat sebuah aliran air terjun dan diketahui aliran tersebut berasal dari sungai yang ada di atas tebing sehingga dinamakan Curug Batu Templek.
- City: Bandung
- Price: 5000
- Rating: 4.10

- Recommended Category: 416, Estimated Rating: 3.95
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.91
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.91
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20


**Top Recommendations for User 248**
- Recommended Category: 416, Estimated Rating: 4.30
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 136, Estimated Rating: 4.27
- Places in category '136' with similar rating ---
- Place: 136
- Place: Grojogan Watu Purbo Bangunrejo
- Description: Objek wisata itu tak adalah Grojogan Watu Purbo yang berada di Bangunrejo, Merdikorejo, Kecamatan Tempel. Objek wisata itu sekitar setahun terakhir cukup populer di kalangan wisatawan karena memiliki pemandangan eksotis berupa air terjun yang memiliki enam tingkatan. Wisatawan yang datang rata-rata menjadikan air terjun itu sebagai latar untuk swafoto karena pemandangannya yang dinilai instagramable. Grojokan Watu Purbo ini tepatnya berlokasi di aliran Kali Krisak, yang merupakan jalur dari lahar dingin yang mengalir dari Gunung Merapi. Pemandangan kawasan ini eksotis karena dikepung pepohonan asri serta hamparan sawah. Munculnya air terjun atau grojogan ini berasal dari enam dam dengan ketinggian bervariasi tak lebih dari 10 meter.
- City: Yogyakarta
- Price: 10000
- Rating: 4.50

- Recommended Category: 115, Estimated Rating: 4.26
- Places in category '115' with similar rating ---
- Place: 115
- Place: Monumen Sanapati
- Description: Monumen Sanapati dibangun untuk memeringati 50 tahun persandian Indonesia di Yogyakarta. Persandian di Yogyakarta memiliki andil yang besar dalam mempertahankan kemerdekaan Negara Indonesia. Tercatat pada peristiwa Serangan Umum 1 Maret 1949, sandi buatan Roebiono masih digunakan untuk menyebarkan pesan pada dunia bahwa Indonesia masih ada dan masih bisa melawan. Selanjutnya, peristiwa tersebut dikenal dengan Peristiwa 6 jam di Yogyakarta. Monumen Sanapati terletak di kawasan Kotabaru, tepatnya di tengah persimpangan Jalan Abu Bakar Ali, di taman depan Gereja Santo Antonius. Bangunan Monumen Sanapati berbentuk segitiga berujung lancip menyerupai piramida dengan tinggi 3 meter dan lebar 2,5 meter. Monumen tersebut didesain oleh Drs Kasman dan diresmikan oleh Menteri Sekretaris Negara Republik Indonesia saat itu, Moerdiono, pada 4 April 1996, tepat pada Hari Persandian Indonesia. Pembangunan monument membutuhkan waktu satu bulan dan memakan biaya sebesar 25 juta rupiah.
- City: Yogyakarta
- Price: 15000
- Rating: 4.30

- Recommended Category: 100, Estimated Rating: 4.19
- Places in category '100' with similar rating ---
- Place: 100
- Place: Taman Budaya Yogyakarta
- Description: Taman Budaya Yogyakarta (TBY) (Hanacaraka:Í¶†Í¶©Í¶§ÍßÄ‚ÄãÍ¶ßÍ¶∏Í¶¢Í¶™‚ÄãÍ¶îÍ¶™Í¶∫Í¶¥Í¶íÍ¶æÍ¶èÍ¶ÇÍ¶†, bahasa Jawa: Taman Budaya Ngayogyakarta) adalah sarana wisata yang terletak di Jalan Sri Wedani No 1, Yogyakarta. TBY memiliki kompleks gedung yang berfungsi sebagai tempat pameran, pertunjukan, dan berbagai kegiatan seni lainnya. TBY merupakan Unit Pelaksana Teknis Dinas (UPTD) pada Dinas Kebudayaan Provinsi Daerah Istimewa Yogyakarta. Fungsi dari TBY adalah sebagai pusat budaya termasuk di dalamnya pengembangan dan pengolahan pusat dokumentasi, etalase, dan informasi seni budaya dan pariwisata._x000D_
- City: Yogyakarta
- Price: 0
- Rating: 4.50

- Recommended Category: 157, Estimated Rating: 4.19
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40


**Top Recommendations for User 249**
- Recommended Category: 416, Estimated Rating: 3.27
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.17
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 3.16
- Places in category '79' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.11
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.10
- Places in category '132' with similar rating ---
- - No similar places found


**Top Recommendations for User 250**
- Recommended Category: 157, Estimated Rating: 4.27
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 44, Estimated Rating: 4.22
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 4.19
- Places in category '79' with similar rating ---
- Place: 79
- Place: Taman Spathodea
- Description: Objek Wisata Taman Spathodea di Jagakarsa DKI Jakarta Selatan Jakarta adalah salah satu tempat wisata yang berada di Jl. Kebagusan Raya, Kecamatan Jagakarsa, Kota Jakarta Selatan, Daerah Khusus Ibukota Jakarta, Indonesia. Objek Wisata Taman Spathodea di Jagakarsa DKI Jakarta Selatan Jakarta adalah tempat wisata yang ramai dengan wisatawan pada hari biasa maupun saat liburan. Yang menjadi daya tarik Taman Spathodea memiliki banyak jenis tanaman yang ditanam baik pepohonan dan bunga-bunga yang menghiasi area taman. Ada danau kecil yang berada ditengah taman akan membuat nyaman pengunjung. Banyak Fasilitas umum juga tersedia di Taman Spathodea seperti misalnya lintasan jogging, kolam ikan, outdoor gym, taman bermain untuk anak-anak, toilet, parkiran yang cukup luas dan tempat duduk yang cukup nyaman. Taman Spathodea ini hanya buka mulai pukul 05.00 ‚Äì 18.00 WIB jadi kalo sudah mulai malam taman ini akan ditutup untuk umum.
- City: Jakarta
- Price: 0
- Rating: 4.60

- Recommended Category: 416, Estimated Rating: 4.19
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 230, Estimated Rating: 4.19
- Places in category '230' with similar rating ---
- Place: 230
- Place: Stone Garden Citatah
- Description: Stone  Garden, adalah sebutan nama untuk hamparan tanah yang diisi oleh  formasi batuan tak beraturan yang indah dan membentuk taman alam.Bandung  memiliki banyak wisata alam yang sangat luar biasa. Ternyata Bandung  memiliki wisata alam gua, yaitu Gua Pawon dan wisata Stone Garden. Gua  Pawon merupakan gua yang terbentuk pada zaman purba.Gua ini digunakan  oleh manusia purba sebagai tempat berlindung. Dan uniknya di atas Gua  Pawon terdapat hamparan bebatuan yang sangat indah dengan pemandangan  luar biasa dikenal dengan nama Stone Garden -Taman Batu. Tidak kalah  indah dengan wisata alam Tangkuban Perahu, Kawah Putih Ciwidey, dan Situ  Patengan merupakan kawasan wisata alam yang sering dikunjungi oleh  banyak wisatawan lokal maupun mancanegara. _x000D_
- City: Bandung
- Price: 30000
- Rating: 4.40


**Top Recommendations for User 251**
- Recommended Category: 232, Estimated Rating: 3.93
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.68
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 164, Estimated Rating: 3.57
- Places in category '164' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.57
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 182, Estimated Rating: 3.55
- Places in category '182' with similar rating ---
- - No similar places found


**Top Recommendations for User 252**
- Recommended Category: 322, Estimated Rating: 3.95
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 44, Estimated Rating: 3.95
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.94
- Places in category '314' with similar rating ---
- Place: 314
- Place: Tafso Barn
- Description: Nama Punclut mungkin sudah cukup akrab di telinga wisatawan. Kawasan dataran tinggi di Bandung yang belakangan populer sebagai destinasi wisata. Selain menawarkan pemandangan yang indah, kawasan ini juga memiliki banyak tempat wisata untuk rekreasi. Salah satunya adalah kawasan wisata sekaligus restoran Tafso dan Boda Barn atau sering dikenal juga dengan Tafso Barn. Tempat rekreasi ini memadukan tempat makan dengan nuansa taman dan gudang yang unik. Pemandangan perbukitan yang dipotong garis horison menjadi incaran utama pengunjung. Sangat tepat nongkrong dan bersantai ketika berlibur di Bandung.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 416, Estimated Rating: 3.94
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 377, Estimated Rating: 3.90
- Places in category '377' with similar rating ---
- - No similar places found


**Top Recommendations for User 253**
- Recommended Category: 416, Estimated Rating: 3.61
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.58
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.58
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 3.56
- Places in category '83' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.54
- Places in category '254' with similar rating ---
- - No similar places found


**Top Recommendations for User 254**
- Recommended Category: 279, Estimated Rating: 4.00
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.91
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 3.66
- Places in category '401' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.65
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.65
- Places in category '314' with similar rating ---
- - No similar places found


**Top Recommendations for User 255**
- Recommended Category: 139, Estimated Rating: 3.92
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.83
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 407, Estimated Rating: 3.82
- Places in category '407' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.81
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.81
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 256**
- Recommended Category: 97, Estimated Rating: 4.00
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.93
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.91
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.79
- Places in category '314' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 3.79
- Places in category '422' with similar rating ---
- - No similar places found


**Top Recommendations for User 257**
- Recommended Category: 416, Estimated Rating: 3.81
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.70
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.63
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 251, Estimated Rating: 3.58
- Places in category '251' with similar rating ---
- - No similar places found

- Recommended Category: 90, Estimated Rating: 3.56
- Places in category '90' with similar rating ---
- - No similar places found


**Top Recommendations for User 258**
- Recommended Category: 138, Estimated Rating: 3.87
- Places in category '138' with similar rating ---
- - No similar places found

- Recommended Category: 235, Estimated Rating: 3.76
- Places in category '235' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.76
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.75
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 295, Estimated Rating: 3.70
- Places in category '295' with similar rating ---
- - No similar places found


**Top Recommendations for User 259**
- Recommended Category: 322, Estimated Rating: 3.84
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 279, Estimated Rating: 3.83
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.76
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.66
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.62
- Places in category '321' with similar rating ---
- - No similar places found


**Top Recommendations for User 260**
- Recommended Category: 1, Estimated Rating: 4.02
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 261, Estimated Rating: 3.83
- Places in category '261' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.83
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.82
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.82
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 261**
- Recommended Category: 416, Estimated Rating: 3.81
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.68
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.66
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 226, Estimated Rating: 3.46
- Places in category '226' with similar rating ---
- - No similar places found

- Recommended Category: 253, Estimated Rating: 3.45
- Places in category '253' with similar rating ---
- - No similar places found


**Top Recommendations for User 262**
- Recommended Category: 52, Estimated Rating: 4.12
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 227, Estimated Rating: 3.93
- Places in category '227' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.91
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.89
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 171, Estimated Rating: 3.88
- Places in category '171' with similar rating ---
- - No similar places found


**Top Recommendations for User 263**
- Recommended Category: 249, Estimated Rating: 3.76
- Places in category '249' with similar rating ---
- Place: 249
- Place: Upside Down World Bandung
- Description: Upside Down World Bandung pertama kali dibuka pada 10 Oktober 2016 dengan konsep yang sangat unik. Semua furnitur yang berada di Upside Down World ini dipasang terbalik seolah pengunjung berada di langit-langit ruangan. Dengan konsep rumah terbaik tempat wisata ini memiliki keunikan tersendiri. Pengunjung yang datang ke sini kebanyakan para kawula muda yang ingin berfoto dengan latar yang tak biasa. Di dalam bangunan ini terdapat sepuluh ruangan dengan tema yang berbeda-beda layaknya di dalam sebuah rumah. Ada ruang tamu, kamar tidur anak, ruang keluarga, ruang santai, dapur, kamar mandi, dan lain sebagainya. Pengunjung bisa mengambil gambar pada semua ruangan dan hasil foto dapat dicetak di layanan cetak foto. Jika pengunjung merasa tak biasa dan bingung harus bergaya foto seperti apa, maka akan ada petugas yang dapat mengarahkan gayanya. Atau dengan melihat hasil foto para pengunjung sebelumnya yang dipasang di ruangan itu._x000D_
- City: Bandung
- Price: 100000
- Rating: 4.00

- Recommended Category: 419, Estimated Rating: 3.75
- Places in category '419' with similar rating ---
- - No similar places found

- Recommended Category: 60, Estimated Rating: 3.75
- Places in category '60' with similar rating ---
- - No similar places found

- Recommended Category: 98, Estimated Rating: 3.74
- Places in category '98' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 3.74
- Places in category '79' with similar rating ---
- - No similar places found


**Top Recommendations for User 264**
- Recommended Category: 416, Estimated Rating: 3.81
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.75
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.68
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.66
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.64
- Places in category '44' with similar rating ---
- - No similar places found


**Top Recommendations for User 265**
- Recommended Category: 139, Estimated Rating: 3.67
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.56
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.50
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.47
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 263, Estimated Rating: 3.47
- Places in category '263' with similar rating ---
- - No similar places found


**Top Recommendations for User 266**
- Recommended Category: 322, Estimated Rating: 3.97
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 279, Estimated Rating: 3.91
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 3.89
- Places in category '83' with similar rating ---
- - No similar places found

- Recommended Category: 125, Estimated Rating: 3.89
- Places in category '125' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.88
- Places in category '1' with similar rating ---
- - No similar places found


**Top Recommendations for User 267**
- Recommended Category: 139, Estimated Rating: 3.88
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 28, Estimated Rating: 3.81
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.79
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.68
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 230, Estimated Rating: 3.65
- Places in category '230' with similar rating ---
- - No similar places found


**Top Recommendations for User 268**
- Recommended Category: 416, Estimated Rating: 4.05
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 139, Estimated Rating: 4.00
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.84
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 138, Estimated Rating: 3.82
- Places in category '138' with similar rating ---
- - No similar places found

- Recommended Category: 136, Estimated Rating: 3.73
- Places in category '136' with similar rating ---
- - No similar places found


**Top Recommendations for User 269**
- Recommended Category: 1, Estimated Rating: 3.46
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 387, Estimated Rating: 3.46
- Places in category '387' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.38
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.36
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.31
- Places in category '44' with similar rating ---
- - No similar places found


**Top Recommendations for User 270**
- Recommended Category: 134, Estimated Rating: 3.53
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 399, Estimated Rating: 3.47
- Places in category '399' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.46
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 332, Estimated Rating: 3.46
- Places in category '332' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.43
- Places in category '44' with similar rating ---
- - No similar places found


**Top Recommendations for User 271**
- Recommended Category: 416, Estimated Rating: 4.01
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 139, Estimated Rating: 4.01
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 253, Estimated Rating: 3.84
- Places in category '253' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.83
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 214, Estimated Rating: 3.82
- Places in category '214' with similar rating ---
- - No similar places found


**Top Recommendations for User 272**
- Recommended Category: 322, Estimated Rating: 3.86
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 112, Estimated Rating: 3.82
- Places in category '112' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.78
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.72
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.70
- Places in category '139' with similar rating ---
- - No similar places found


**Top Recommendations for User 273**
- Recommended Category: 416, Estimated Rating: 4.07
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 157, Estimated Rating: 3.87
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.84
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 94, Estimated Rating: 3.83
- Places in category '94' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.79
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 274**
- Recommended Category: 416, Estimated Rating: 4.31
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 138, Estimated Rating: 4.10
- Places in category '138' with similar rating ---
- Place: 138
- Place: Jogja Exotarium
- Description: Di Yogyakarta, tepatnya di Sleman, ada satu tempat wisata edukasi yang patut dikunjungi. Namanya adalah Jogja Exotarium ‚Äî terdengar unik, kan? Namun sebenarnya, tempat ini merupakan taman hewan berskala kecil. Koleksi hewannya beragam dan bisa diajak berinteraksi secara langsung
- City: Yogyakarta
- Price: 20000
- Rating: 4.40

- Recommended Category: 44, Estimated Rating: 4.04
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 91, Estimated Rating: 3.98
- Places in category '91' with similar rating ---
- - No similar places found

- Recommended Category: 200, Estimated Rating: 3.98
- Places in category '200' with similar rating ---
- - No similar places found


**Top Recommendations for User 275**
- Recommended Category: 157, Estimated Rating: 3.08
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.07
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 431, Estimated Rating: 3.07
- Places in category '431' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.06
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.04
- Places in category '416' with similar rating ---
- - No similar places found


**Top Recommendations for User 276**
- Recommended Category: 115, Estimated Rating: 3.55
- Places in category '115' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.52
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 208, Estimated Rating: 3.46
- Places in category '208' with similar rating ---
- - No similar places found

- Recommended Category: 118, Estimated Rating: 3.38
- Places in category '118' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.38
- Places in category '279' with similar rating ---
- - No similar places found


**Top Recommendations for User 277**
- Recommended Category: 279, Estimated Rating: 4.20
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 4.16
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 28, Estimated Rating: 3.93
- Places in category '28' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.91
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.83
- Places in category '134' with similar rating ---
- - No similar places found


**Top Recommendations for User 278**
- Recommended Category: 52, Estimated Rating: 3.46
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.44
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.42
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.36
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 117, Estimated Rating: 3.33
- Places in category '117' with similar rating ---
- - No similar places found


**Top Recommendations for User 279**
- Recommended Category: 416, Estimated Rating: 3.76
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.69
- Places in category '314' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.61
- Places in category '132' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.56
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 122, Estimated Rating: 3.54
- Places in category '122' with similar rating ---
- - No similar places found


**Top Recommendations for User 280**
- Recommended Category: 416, Estimated Rating: 3.99
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 399, Estimated Rating: 3.64
- Places in category '399' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.56
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.56
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 420, Estimated Rating: 3.55
- Places in category '420' with similar rating ---
- - No similar places found


**Top Recommendations for User 281**
- Recommended Category: 322, Estimated Rating: 3.70
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.61
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.60
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.59
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.52
- Places in category '1' with similar rating ---
- - No similar places found


**Top Recommendations for User 282**
- Recommended Category: 416, Estimated Rating: 3.72
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.71
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 321, Estimated Rating: 3.68
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.67
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 224, Estimated Rating: 3.67
- Places in category '224' with similar rating ---
- - No similar places found


**Top Recommendations for User 283**
- Recommended Category: 139, Estimated Rating: 3.79
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.78
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 164, Estimated Rating: 3.78
- Places in category '164' with similar rating ---
- - No similar places found

- Recommended Category: 125, Estimated Rating: 3.73
- Places in category '125' with similar rating ---
- - No similar places found

- Recommended Category: 112, Estimated Rating: 3.68
- Places in category '112' with similar rating ---
- - No similar places found


**Top Recommendations for User 284**
- Recommended Category: 279, Estimated Rating: 3.68
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.62
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.62
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.58
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.53
- Places in category '300' with similar rating ---
- - No similar places found


**Top Recommendations for User 285**
- Recommended Category: 416, Estimated Rating: 4.17
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 134, Estimated Rating: 4.08
- Places in category '134' with similar rating ---
- Place: 134
- Place: Desa Wisata Gamplong
- Description: Desa Wisata Gamplong adalah desa wisata kerajinan tenun yang berada di Padukuhan Gamplong Desa Sumber Rahayu Kecamatan Moyudan Kabupaten Sleman, Yogyakarta. Desa wisata yang terletak di sebelah barat Kota Yogyakarta ini cukup menarik untuk disinggahi wisatawan terlebih karena masih adanya industri kerajinan tenun tradisional dengan menggunakan Alat Tenun Bukan Mesin (ATBM). Dengan ATBM, masyarakat perajin Gamplong mampu menghasilkan kain tenun sebagai bahan stagen (kain panjang untuk melilit bagian perut wanita). Selain kerajinan tenun, perajin juga mampu memproduksi kerajinan anyaman untuk suvenir.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 172, Estimated Rating: 3.96
- Places in category '172' with similar rating ---
- - No similar places found

- Recommended Category: 80, Estimated Rating: 3.92
- Places in category '80' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.91
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 286**
- Recommended Category: 97, Estimated Rating: 3.75
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.72
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.68
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.65
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.64
- Places in category '157' with similar rating ---
- - No similar places found


**Top Recommendations for User 287**
- Recommended Category: 52, Estimated Rating: 3.89
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.86
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.84
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.77
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 79, Estimated Rating: 3.74
- Places in category '79' with similar rating ---
- - No similar places found


**Top Recommendations for User 288**
- Recommended Category: 52, Estimated Rating: 4.06
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.90
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.85
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 321, Estimated Rating: 3.79
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.70
- Places in category '254' with similar rating ---
- - No similar places found


**Top Recommendations for User 289**
- Recommended Category: 139, Estimated Rating: 3.71
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.62
- Places in category '232' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.58
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 353, Estimated Rating: 3.54
- Places in category '353' with similar rating ---
- - No similar places found

- Recommended Category: 97, Estimated Rating: 3.53
- Places in category '97' with similar rating ---
- - No similar places found


**Top Recommendations for User 290**
- Recommended Category: 422, Estimated Rating: 3.83
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.81
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 279, Estimated Rating: 3.78
- Places in category '279' with similar rating ---
- - No similar places found

- Recommended Category: 1, Estimated Rating: 3.68
- Places in category '1' with similar rating ---
- - No similar places found

- Recommended Category: 314, Estimated Rating: 3.66
- Places in category '314' with similar rating ---
- - No similar places found


**Top Recommendations for User 291**
- Recommended Category: 122, Estimated Rating: 3.59
- Places in category '122' with similar rating ---
- - No similar places found

- Recommended Category: 416, Estimated Rating: 3.58
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.57
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 3.54
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.53
- Places in category '132' with similar rating ---
- - No similar places found


**Top Recommendations for User 292**
- Recommended Category: 416, Estimated Rating: 3.96
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.92
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.91
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 401, Estimated Rating: 3.79
- Places in category '401' with similar rating ---
- - No similar places found

- Recommended Category: 230, Estimated Rating: 3.60
- Places in category '230' with similar rating ---
- - No similar places found


**Top Recommendations for User 293**
- Recommended Category: 97, Estimated Rating: 3.81
- Places in category '97' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.80
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.76
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 422, Estimated Rating: 3.71
- Places in category '422' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.67
- Places in category '300' with similar rating ---
- - No similar places found


**Top Recommendations for User 294**
- Recommended Category: 146, Estimated Rating: 3.92
- Places in category '146' with similar rating ---
- - No similar places found

- Recommended Category: 322, Estimated Rating: 3.87
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 433, Estimated Rating: 3.82
- Places in category '433' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 3.79
- Places in category '52' with similar rating ---
- - No similar places found

- Recommended Category: 401, Estimated Rating: 3.78
- Places in category '401' with similar rating ---
- - No similar places found


**Top Recommendations for User 295**
- Recommended Category: 416, Estimated Rating: 3.94
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 139, Estimated Rating: 3.70
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 254, Estimated Rating: 3.60
- Places in category '254' with similar rating ---
- - No similar places found

- Recommended Category: 183, Estimated Rating: 3.51
- Places in category '183' with similar rating ---
- - No similar places found

- Recommended Category: 253, Estimated Rating: 3.50
- Places in category '253' with similar rating ---
- - No similar places found


**Top Recommendations for User 296**
- Recommended Category: 322, Estimated Rating: 3.72
- Places in category '322' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.67
- Places in category '132' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 3.65
- Places in category '83' with similar rating ---
- - No similar places found

- Recommended Category: 134, Estimated Rating: 3.60
- Places in category '134' with similar rating ---
- - No similar places found

- Recommended Category: 232, Estimated Rating: 3.57
- Places in category '232' with similar rating ---
- - No similar places found


**Top Recommendations for User 297**
- Recommended Category: 416, Estimated Rating: 4.29
- Places in category '416' with similar rating ---
- Place: 416
- Place: Keraton Surabaya
- Description: Kawasan yang berjuluk Kampung Keraton ini terdiri dari empat gang. Gang ini rata-rata berukuran sempit, lebarnya 2 hingga 3 meter. Saking sempitnya, pengendara sepeda motor terkadang turun dari kendaraan dan harus berjalan kaki. Sementara, panjang jalan sekitar 200 meter. Memang, tak banyak sisa peninggalan Keraton Surabaya ini. Yang tampak kini adalah sisa bangunan peninggalan kolonial Belanda. Kampung Keraton saat ini merupakan salah satu kawasan perdagangan yang sangat ramai di Surabaya, utamanya di Jalan Kramat Gantung. Segala macam kebutuhan rumah tangga dijual di jalan ini.
- City: Surabaya
- Price: 0
- Rating: 4.40

- Recommended Category: 322, Estimated Rating: 4.00
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 157, Estimated Rating: 3.96
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 230, Estimated Rating: 3.90
- Places in category '230' with similar rating ---
- - No similar places found

- Recommended Category: 132, Estimated Rating: 3.90
- Places in category '132' with similar rating ---
- - No similar places found


**Top Recommendations for User 298**
- Recommended Category: 97, Estimated Rating: 4.38
- Places in category '97' with similar rating ---
- Place: 97
- Place: Monumen Yogya Kembali
- Description: Museum Monumen Yogya Kembali (bahasa Jawa: Í¶©Í¶∫Í¶¥Í¶§Í¶∏Í¶©Í¶∫Í¶§ÍßÄ‚ÄãÍ¶™Í¶∫Í¶¥Í¶íÍ¶æ‚ÄãÍ¶èÍ¶ºÍ¶©ÍßÄÍ¶ßÍ¶≠Í¶∂, translit. Monum√®n Yogya Kembali) biasa dikenal sebagai Monumen Jogja Kembali disingkat Monjali adalah sebuah museum sejarah perjuangan kemerdekaan Indonesia yang ada di Daerah Istimewa Yogyakarta dan dikelola oleh Kementerian Pariwisata dan Ekonomi Kreatif. Museum yang berada di bagian utara kota ini banyak dikunjungi oleh para pelajar dalam acara darmawisata.\n\nMuseum monumen dengan bentuk kerucut ini terdiri dari 3 lantai dan dilengkapi dengan ruang perpustakaan serta ruang serbaguna. Pada rana pintu masuk dituliskan sejumlah 422 nama pahlawan yang gugur di daerah Wehrkreise III (RIS) antara tanggal 19 Desember 1948 sampai dengan 29 Juni 1949. Dalam 4 ruang museum di lantai 1 terdapat benda-benda koleksi: relief, replika, foto, dokumen, heraldika, berbagai jenis senjata, bentuk evokatif dapur umum dalam suasana perang kemerdekaan 1945-1949. Tandu dan dokar (kereta kuda) yang pernah dipergunakan oleh Panglima Besar Jenderal Soedirman juga disimpan di sini (di ruang museum nomor 2). Monumen Yogya Kembali beralamat di Jl. Ring Road Utara, Kabupaten Sleman, Daerah Istimewa Yogyakarta.
- City: Yogyakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 52, Estimated Rating: 4.19
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 322, Estimated Rating: 4.18
- Places in category '322' with similar rating ---
- Place: 322
- Place: Bukit Jamur
- Description: Bukit Jamur Ciwidey adalah satu dari sekian banyak pesona wisata alam yang berada di Kecamatan Ciwidey, Kabupaten Bandung, Provinsi Jawa barat. Bukit Jamur Ciwidey memang memiliki khas wisata Ciwidey, yaitu objek wisata di tengah hamparan kebun teh, dengan udara yang sangat sejuk.Suasana saat kabut turun di area perkebunan membuat suasana menjadi romatis. Bukit Jamur Ciwidey suasananya cocok juga bagi jomblowan ‚Äì jomblowati, banyak spot dengan latar alam yang indah, yang dimungkinkan mampu melupakan sang mantan.
- City: Bandung
- Price: 0
- Rating: 4.20

- Recommended Category: 157, Estimated Rating: 4.15
- Places in category '157' with similar rating ---
- Place: 157
- Place: Pantai Baron
- Description: Pantai Baron adalah salah satu objek wisata berupa pantai yang terletak di Desa Kemadang, Kecamatan Tanjungsari, Kabupaten Gunungkidul. Lokasi Pantai Baron dapat ditempuh 40 km daeri pusat kota Yogyakarta. Asal mula nama Pantai Baron berasal dari nama seorang bangsawan asal Belanda yang bernama Baron Skeber. Bangsawan tersebut pernah mendaratkan kapalnya di pantai selatan tepatnya di pantai yang saat ini terkenal dengan sebutan Pantai Baron. Jalan menuju objek wisata Pantai Baron cukup baik untuk dilalui kendaraan pribadi, sepeda motor dan bus.Objek wisata Pantai Baron merupakan pantai yang membentuk cekungan. Seperti pantai lainnya, di Pantai Baron tersedia aneka ikan laut beserta olahannya. Ikan yang biasanya dijual di Pantai Baron adalah udang windu, kakap, bawal putih dan tongkol. Pantai Baron memiliki fasilitas berupa tempat pelelangan ikan, wahana permainan anak-anak, perahu bermesin, dan toko cenderamata. Buah sirkaya, pisang tanduk, sirsak, dan berbagai macam cenderamata yang terbuat dari kerang laut. Cenderamata berbahan kerang yang banyak dijual di Pantai Baron adalah bros, tirai kerang, lampu hias, cermin berhias karang, figura, dan aneka karakter hewan yang juga terbuat dari kerang laut.Upacara sedekah laut adalah upacara yang masih sering dilakukan oleh masyarakat Gunungkidul. Pantai Baron adalah salah satu tempat untuk menyelenggarakan upacara sedekah laut tersebut. Upacara sedekah laut diselenggarakan setiap tanggal satu Syuro dalam penanggalan Jawa. Upacara sedekah laut dilakukan sebagai ungkapan rasa syukur penduduk setempat atas melimpahnya tangkapan ikan di Pantai Baron. Awalnya, sebagaian besar penduduk di sekitar Pantai Baron bukanlah nelayan melainkan petani yang mengolah kebun. Suatu ketika ada seseorang yang memulai menangkap ikan di pinggiran pantai dan mendapatkan banyak ikan, kemudian banyak penduduk yang mengikuti orang tersebut untuk menangkap ikan. Semakin lama, ikan di pinggir pantai semakin sedikit kemudian penduduk mencoba menangkap ikan ketengah laut menggunakan rakit kayu. Adanya kapal di pantai ini karena suatu ketika terjadi tragedi nelayan yang saat menangkap ikan di tengah laut digigit oleh ikan hiu maka kabar ini pun tersiar di mana-mana dan menjadikan pemerintah menyumbangkan perahu untuk nelayan Pantai Baron.Keunikan Pantai Baron adalah adanya sunga bawah tanah yang mengalir cukup deras ke arah lautan. Sungai bawah tanah tersebut mengalir ke arah laut dan membentuk sebuah sungai. Uniknya sungai bawah tanah yang ada di Pantai Baron adalah rasa airnya yang tawar meskipun berada sangat dekat dengan laut. Pengunjung yang tidak berani bermain dan berenang di laut dapat bermain air dan berenang dialiran sungai bawah tanah tersebut. Pemandangan lain yang ada di Pantai Baron adalah sebuah bukit yang berada di sekitar pantai. Pengunjung dapat menikmati keindahan pantai dari atas bukit tersebut.
- City: Yogyakarta
- Price: 10000
- Rating: 4.40

- Recommended Category: 300, Estimated Rating: 4.15
- Places in category '300' with similar rating ---
- Place: 300
- Place: Sanghyang Heuleut
- Description: Danau yang satu ini memiliki air jernih bernuansa kehijauan yang berpadu indah dengan tebing batu berukuran besar di sekelilingnya. Di sela-sela bebatuan, kamu bisa menemukan pepohonan dan rerumputan hijau yang menyegarkan mata. Bebatuannya yang tinggi itu bahkan kerap digunakan sebagai tempat para traveler untuk meloncat indah menuju danau purba tersebut. Danau Sanghyang Heuleut memiliki kedalaman sekitar tiga meter. Jadi, sebelum kamu masuk ke dalam danau, pastikan kamu bisa berenang agar tidak tenggelam, atau setidaknya telah memiliki perlengkapan seperti jaket pelampung. _x000D_
- City: Bandung
- Price: 10000
- Rating: 4.40


**Top Recommendations for User 299**
- Recommended Category: 416, Estimated Rating: 3.71
- Places in category '416' with similar rating ---
- - No similar places found

- Recommended Category: 157, Estimated Rating: 3.66
- Places in category '157' with similar rating ---
- - No similar places found

- Recommended Category: 44, Estimated Rating: 3.57
- Places in category '44' with similar rating ---
- - No similar places found

- Recommended Category: 83, Estimated Rating: 3.56
- Places in category '83' with similar rating ---
- - No similar places found

- Recommended Category: 136, Estimated Rating: 3.55
- Places in category '136' with similar rating ---
- - No similar places found


**Top Recommendations for User 300**
- Recommended Category: 139, Estimated Rating: 4.12
- Places in category '139' with similar rating ---
- - No similar places found

- Recommended Category: 52, Estimated Rating: 4.10
- Places in category '52' with similar rating ---
- Place: 52
- Place: Kampung Cina
- Description: KAMPUNG China adalah hunian dan kawasan perdagangan di Cibubur, Jakarta Timur. Wilayah ini berdiri berkat kerja sama antara Pemerintah Kota Jakarta Timur dengan perusahaan asing untuk menyulap area 300 hektare kawasan wisata Cibubur menjadi kota wisata, mandiri dan town center. Mengutip buku Ensiklopedia Jakarta, kampung itu mulai beroperasi pada tanggal 14 September 2002. Bangunan, nuansa alam, sampai pernak-pernik yang dijual masih berbau kesenian China. Semua produk yang dijajahkan pun berasal dari negeri tirai bambu tersebut. Wilayah itu akan menjadi tujuan wisata pada setiap perayaan hari raya Imlek._x000D_
- City: Jakarta
- Price: 15000
- Rating: 4.50

- Recommended Category: 254, Estimated Rating: 3.95
- Places in category '254' with similar rating ---
- Place: 254
- Place: Teras Cikapundung BBWS
- Description: Teras Cikapundung Bandung sebelumnya merupakan daerah yang kumuh dan tak terurus yang berhasil diubah menjadi tempat wisata. Cikapundung adalah sungai yang membelah Kota Bandung yang dimulai dari daerah Bandung Utara dan bermuara di Sungai Citarum, Bandung Selatan. Sebelumnya, taman ini merupakan tempat berkumpulnya orang-orang kreatif seperti para seniman yang ada di Bandung. Terdapat beberapa sanggar seni dan ramainya warung makan di tempat ini, sampai pada akhirnya terlantar dan dibiarkan kotor. Kini Teras Cikapundung Bandung telah berubah menjadi tempat wisata terbuka untuk umum dengan konsep ekologi dan urban. Berbagai fasilitas menarik tersedia di Teras Cikapundung dan bisa digunakan pada saat rekreasi ke obyek wisata yang satu ini. Dilengkapi dengan kolam ikan dan lampu-lampu, membuat taman ini sangat cantik dikunjungi pada malam hari. Terlebih suasananya yang ramai membuat para pengunjung lupa bahwa dulunya tempat ini pernah dijuluki sebagai ‚Äútempat jin buang anak‚Äù.
- City: Bandung
- Price: 0
- Rating: 4.30

- Recommended Category: 321, Estimated Rating: 3.90
- Places in category '321' with similar rating ---
- - No similar places found

- Recommended Category: 300, Estimated Rating: 3.88
- Places in category '300' with similar rating ---
- - No similar places found


## Method 2

sums the NAN values column-wise, giving you the total number of missing values in each column of the DataFrame
```
User_Id     0
Location    0
Age         0
dtype: int64

Place_Id          0
Place_Name        0
Description       0
Category          0
City              0
Price             0
Rating            0
Time_Minutes    232

Coordinate        0
Lat               0
Long              0
Unnamed: 11     437
Unnamed: 12       0
dtype: int64

User_Id          0
Place_Id         0
Place_Ratings    0
dtype: int64

Place_Name
Keraton Surabaya                         3.933333
Puncak Gunung Api Purba - Nglanggeran    3.882353
Kampung Cina                             3.842105
Bukit Jamur                              3.793103
Teras Cikapundung BBWS                   3.789474
Name: Place_Ratings, dtype: float64

City
Yogyakarta    3.104986
Bandung       3.079022
Surabaya      3.078035
Semarang      3.035850
Jakarta       3.007361

Name: Place_Ratings, dtype: float64
Category
Taman Hiburan    3.118386
Tempat Ibadah    3.086387
Cagar Alam       3.081352
Budaya           3.032270
Bahari           3.011194

Name: Place_Ratings, dtype: float64
Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    1.4387  1.4106  1.4313  1.4194  1.4093  1.4218  0.0115  
MAE (testset)     1.2394  1.2094  1.2306  1.2187  1.2110  1.2218  0.0115  
Fit time          0.03    0.03    0.03    0.03    0.03    0.03    0.00    
Test time         0.00    0.00    0.00    0.00    0.00    0.00    0.00    
```
**Top 5 Recommendations for User:** 1
- Bukit Jamur (Predicted Rating: 4.10)
- Keraton Surabaya (Predicted Rating: 4.09)
- Teras Cikapundung BBWS (Predicted Rating: 4.05)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.05)
- Rainbow Garden (Predicted Rating: 4.03)

**Top 5 Recommendations for User:** 2
- Museum Bank Indonesia (Predicted Rating: 3.96)
- Gereja Perawan Maria Tak Berdosa Surabaya (Predicted Rating: 3.94)
- Monumen Yogya Kembali (Predicted Rating: 3.92)
- Kampung Cina (Predicted Rating: 3.85)
- Bukit Jamur (Predicted Rating: 3.82)

**Top 5 Recommendations for User:** 3
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.04)
- Tafso Barn (Predicted Rating: 4.04)
- Trans Studio Bandung (Predicted Rating: 3.99)
- Desa Wisata Gamplong (Predicted Rating: 3.95)
- Bukit Jamur (Predicted Rating: 3.93)

**Top 5 Recommendations for User:** 4
- Keraton Surabaya (Predicted Rating: 4.24)
- Bukit Bintang Yogyakarta (Predicted Rating: 4.17)
- Atlantis Water Adventure (Predicted Rating: 4.12)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.12)
- Jogja Exotarium (Predicted Rating: 4.12)

**Top 5 Recommendations for User:** 5
- Situs Warungboto (Predicted Rating: 4.13)
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 4.13)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.10)
- Curug Tilu Leuwi Opat (Predicted Rating: 4.09)
- Curug Batu Templek (Predicted Rating: 4.07)

**Top 5 Recommendations for User:** 6
- Keraton Surabaya (Predicted Rating: 4.33)
- Monumen Yogya Kembali (Predicted Rating: 4.23)
- Dago Dreampark (Predicted Rating: 4.22)
- Pemandian Air Panas Ciater (Predicted Rating: 4.17)
- Glamping Lakeside Rancabali (Predicted Rating: 4.17)

**Top 5 Recommendations for User:** 7
- Keraton Surabaya (Predicted Rating: 4.69)
- Monumen Nasional (Predicted Rating: 4.52)
- Jogja Exotarium (Predicted Rating: 4.48)
- Selasar Sunaryo Art Space (Predicted Rating: 4.45)
- Dago Dreampark (Predicted Rating: 4.43)

**Top 5 Recommendations for User:** 8
- Monumen Yogya Kembali (Predicted Rating: 4.24)
- Keraton Surabaya (Predicted Rating: 4.22)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.19)
- Glamping Lakeside Rancabali (Predicted Rating: 4.12)
- Desa Wisata Gamplong (Predicted Rating: 4.10)

**Top 5 Recommendations for User:** 9
- Monumen Yogya Kembali (Predicted Rating: 4.45)
- Bukit Jamur (Predicted Rating: 4.42)
- Keraton Surabaya (Predicted Rating: 4.41)
- Pantai Baron (Predicted Rating: 4.41)
- Kampung Cina (Predicted Rating: 4.39)

**Top 5 Recommendations for User:** 10
- Keraton Surabaya (Predicted Rating: 4.44)
- Kampung Cina (Predicted Rating: 4.37)
- Pantai Baron (Predicted Rating: 4.34)
- Desa Wisata Gamplong (Predicted Rating: 4.28)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.28)

**Top 5 Recommendations for User:** 11
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.14)
- Bukit Jamur (Predicted Rating: 4.05)
- Atlantis Water Adventure (Predicted Rating: 4.00)
- Teras Cikapundung BBWS (Predicted Rating: 3.96)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.95)

**Top 5 Recommendations for User:** 12
- Jogja Bay Pirates Adventure Waterpark (Predicted Rating: 4.51)
- Atlantis Water Adventure (Predicted Rating: 4.43)
- Air Terjun Kedung Pedut (Predicted Rating: 4.42)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.41)
- Dago Dreampark (Predicted Rating: 4.41)

**Top 5 Recommendations for User:** 13
- Keraton Surabaya (Predicted Rating: 4.34)
- Gereja Perawan Maria Tak Berdosa Surabaya (Predicted Rating: 4.16)
- Trans Studio Bandung (Predicted Rating: 4.12)
- Sanghyang Heuleut (Predicted Rating: 4.10)
- Pintoe Langit Dahromo (Predicted Rating: 4.05)

**Top 5 Recommendations for User:** 14
- Stone Garden Citatah (Predicted Rating: 4.19)
- Upside Down World Bandung (Predicted Rating: 4.19)
- Air Terjun Kedung Pedut (Predicted Rating: 4.18)
- Situs Warungboto (Predicted Rating: 4.18)
- Kampung Wisata Taman Sari (Predicted Rating: 4.18)

**Top 5 Recommendations for User:** 15
- Keraton Surabaya (Predicted Rating: 4.20)
- Taman Kasmaran (Predicted Rating: 4.16)
- Kampung Wisata Taman Sari (Predicted Rating: 4.11)
- Bukit Bintang Yogyakarta (Predicted Rating: 4.10)
- Sanghyang Heuleut (Predicted Rating: 4.08)

**Top 5 Recommendations for User:** 16
- Bukit Jamur (Predicted Rating: 4.29)
- Air Terjun Kedung Pedut (Predicted Rating: 4.28)
- Keraton Surabaya (Predicted Rating: 4.19)
- Monumen Nasional (Predicted Rating: 4.14)
- Taman Hiburan Rakyat (Predicted Rating: 4.07)

**Top 5 Recommendations for User:** 17
- Monumen Jalesveva Jayamahe (Predicted Rating: 4.18)
- Bukit Jamur (Predicted Rating: 4.13)
- Taman Pelangi Yogyakarta (Predicted Rating: 4.12)
- Pantai Watu Kodok (Predicted Rating: 4.10)
- Taman Hiburan Rakyat (Predicted Rating: 4.09)

**Top 5 Recommendations for User:** 18
- Keraton Surabaya (Predicted Rating: 4.32)
- Pantai Baron (Predicted Rating: 4.32)
- Bukit Jamur (Predicted Rating: 4.27)
- Plaza Indonesia (Predicted Rating: 4.21)
- Jogja Exotarium (Predicted Rating: 4.19)

**Top 5 Recommendations for User:** 19
- Keraton Surabaya (Predicted Rating: 4.33)
- Air Terjun Kedung Pedut (Predicted Rating: 4.29)
- Pantai Congot (Predicted Rating: 4.28)
- Teras Cikapundung BBWS (Predicted Rating: 4.26)
- Pantai Baron (Predicted Rating: 4.26)

**Top 5 Recommendations for User:** 20
- Jogja Exotarium (Predicted Rating: 3.96)
- Bukit Jamur (Predicted Rating: 3.91)
- Keraton Surabaya (Predicted Rating: 3.87)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.78)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.77)

**Top 5 Recommendations for User:** 21
- Keraton Surabaya (Predicted Rating: 4.21)
- Alive Museum Ancol (Predicted Rating: 4.10)
- Teras Cikapundung BBWS (Predicted Rating: 4.05)
- Bukit Jamur (Predicted Rating: 4.04)
- Pantai Baron (Predicted Rating: 4.00)

**Top 5 Recommendations for User:** 22
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.06)
- Kampung Cina (Predicted Rating: 4.03)
- Desa Wisata Gamplong (Predicted Rating: 3.99)
- Museum Nasional (Predicted Rating: 3.97)
- Pantai Baron (Predicted Rating: 3.95)

**Top 5 Recommendations for User:** 23
- Bukit Bintang Yogyakarta (Predicted Rating: 4.39)
- Keraton Surabaya (Predicted Rating: 4.27)
- Bukit Jamur (Predicted Rating: 4.18)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 4.18)
- Monumen Yogya Kembali (Predicted Rating: 4.16)

**Top 5 Recommendations for User:** 24
- Situs Warungboto (Predicted Rating: 4.30)
- Curug Batu Templek (Predicted Rating: 4.08)
- Keraton Surabaya (Predicted Rating: 4.06)
- Kampung Cina (Predicted Rating: 3.93)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.87)

**Top 5 Recommendations for User:** 25
- Keraton Surabaya (Predicted Rating: 4.25)
- Bukit Jamur (Predicted Rating: 4.19)
- Pulau Pelangi (Predicted Rating: 4.14)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.11)
- Teras Cikapundung BBWS (Predicted Rating: 4.09)

**Top 5 Recommendations for User:** 26
- Alive Museum Ancol (Predicted Rating: 4.53)
- Keraton Surabaya (Predicted Rating: 4.49)
- Taman Keputran (Predicted Rating: 4.48)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.47)
- Teras Cikapundung BBWS (Predicted Rating: 4.45)

**Top 5 Recommendations for User:** 27
- Teras Cikapundung BBWS (Predicted Rating: 4.17)
- Keraton Surabaya (Predicted Rating: 4.14)
- Glamping Lakeside Rancabali (Predicted Rating: 4.07)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.04)
- Sanghyang Heuleut (Predicted Rating: 4.03)

**Top 5 Recommendations for User:** 28
- Bukit Bintang Yogyakarta (Predicted Rating: 4.61)
- Pantai Baron (Predicted Rating: 4.61)
- Alive Museum Ancol (Predicted Rating: 4.59)
- Waterpark Kenjeran Surabaya (Predicted Rating: 4.54)
- Taman Srigunting (Predicted Rating: 4.49)

**Top 5 Recommendations for User:** 29
- Keraton Surabaya (Predicted Rating: 4.27)
- Sanghyang Heuleut (Predicted Rating: 4.11)
- Kampung Cina (Predicted Rating: 4.08)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.04)
- Dago Dreampark (Predicted Rating: 4.03)

**Top 5 Recommendations for User:** 30
- Pantai Ngrenehan (Predicted Rating: 4.10)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.09)
- Pantai Baron (Predicted Rating: 4.09)
- Keraton Surabaya (Predicted Rating: 4.04)
- Taman Balai Kota Bandung (Predicted Rating: 3.95)

**Top 5 Recommendations for User:** 31
- Sanghyang Heuleut (Predicted Rating: 4.17)
- Monumen Nasional (Predicted Rating: 4.07)
- Museum Tekstil (Predicted Rating: 4.01)
- Pantai Parangtritis (Predicted Rating: 3.95)
- Dago Dreampark (Predicted Rating: 3.95)

**Top 5 Recommendations for User:** 32
- Keraton Surabaya (Predicted Rating: 4.12)
- Rainbow Garden (Predicted Rating: 4.01)
- Monumen Yogya Kembali (Predicted Rating: 3.98)
- Grand Maerakaca (Predicted Rating: 3.97)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.94)

**Top 5 Recommendations for User:** 33
- Desa Wisata Gamplong (Predicted Rating: 4.14)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.96)
- Pantai Baron (Predicted Rating: 3.92)
- Keraton Surabaya (Predicted Rating: 3.92)
- Situs Warungboto (Predicted Rating: 3.91)

**Top 5 Recommendations for User:** 34
- Keraton Surabaya (Predicted Rating: 4.18)
- Monumen Yogya Kembali (Predicted Rating: 4.14)
- Kampung Cina (Predicted Rating: 4.04)
- Sanghyang Heuleut (Predicted Rating: 4.02)
- Jogja Exotarium (Predicted Rating: 3.99)

**Top 5 Recommendations for User:** 35
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.01)
- Teras Cikapundung BBWS (Predicted Rating: 4.01)
- Keraton Surabaya (Predicted Rating: 3.81)
- Curug Batu Templek (Predicted Rating: 3.79)
- Pantai Baron (Predicted Rating: 3.79)

**Top 5 Recommendations for User:** 36
- Pantai Baron (Predicted Rating: 4.19)
- Keraton Surabaya (Predicted Rating: 4.18)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.10)
- Air Terjun Kedung Pedut (Predicted Rating: 4.10)
- Bukit Jamur (Predicted Rating: 4.05)

**Top 5 Recommendations for User:** 37
- Monumen Yogya Kembali (Predicted Rating: 4.41)
- Keraton Surabaya (Predicted Rating: 4.17)
- Monumen Jalesveva Jayamahe (Predicted Rating: 4.16)
- Sanghyang Heuleut (Predicted Rating: 4.14)
- Sumur Gumuling (Predicted Rating: 4.12)

**Top 5 Recommendations for User:** 38
- Keraton Surabaya (Predicted Rating: 4.28)
- Rumah Sipitung (Predicted Rating: 4.28)
- Pantai Baron (Predicted Rating: 4.26)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.24)
- Bukit Jamur (Predicted Rating: 4.23)

**Top 5 Recommendations for User:** 39
- Keraton Surabaya (Predicted Rating: 4.38)
- Kampung Wisata Taman Sari (Predicted Rating: 4.36)
- Monumen Selamat Datang (Predicted Rating: 4.33)
- Pantai Baron (Predicted Rating: 4.29)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.28)

**Top 5 Recommendations for User:** 40
- Rainbow Garden (Predicted Rating: 4.22)
- Sanghyang Heuleut (Predicted Rating: 4.22)
- Sumur Gumuling (Predicted Rating: 4.17)
- Keraton Surabaya (Predicted Rating: 4.14)
- Monumen Sanapati (Predicted Rating: 4.14)

**Top 5 Recommendations for User:** 41
- Bukit Bintang Yogyakarta (Predicted Rating: 4.02)
- Keraton Surabaya (Predicted Rating: 4.01)
- Taman Mundu (Predicted Rating: 3.99)
- Jogja Exotarium (Predicted Rating: 3.93)
- Monumen Yogya Kembali (Predicted Rating: 3.90)

**Top 5 Recommendations for User:** 42
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.26)
- Bukit Jamur (Predicted Rating: 4.21)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 4.16)
- Keraton Surabaya (Predicted Rating: 4.09)
- Bukit Bintang Yogyakarta (Predicted Rating: 4.01)

**Top 5 Recommendations for User:** 43
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.35)
- Keraton Surabaya (Predicted Rating: 4.26)
- Obyek Wisata Goa Kreo (Predicted Rating: 4.24)
- Kampung Cina (Predicted Rating: 4.20)
- Rumah Sipitung (Predicted Rating: 4.07)

**Top 5 Recommendations for User:** 44
- Keraton Surabaya (Predicted Rating: 4.13)
- Jogja Exotarium (Predicted Rating: 4.08)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.06)
- Pemandian Air Panas Ciater (Predicted Rating: 4.06)
- Goa Pindul (Predicted Rating: 4.04)

**Top 5 Recommendations for User:** 45
- Keraton Surabaya (Predicted Rating: 4.43)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.36)
- Teras Cikapundung BBWS (Predicted Rating: 4.29)
- Pasar Kebon Empring Bintaran (Predicted Rating: 4.22)
- Monumen Jalesveva Jayamahe (Predicted Rating: 4.22)

**Top 5 Recommendations for User:** 46
- Glamping Lakeside Rancabali (Predicted Rating: 4.60)
- Keraton Surabaya (Predicted Rating: 4.58)
- Tafso Barn (Predicted Rating: 4.47)
- Monumen Yogya Kembali (Predicted Rating: 4.39)
- Taman Srigunting (Predicted Rating: 4.39)

**Top 5 Recommendations for User:** 47
- Keraton Surabaya (Predicted Rating: 3.49)
- Monumen Nasional (Predicted Rating: 3.40)
- Jogja Bay Pirates Adventure Waterpark (Predicted Rating: 3.31)
- Pantai Watu Kodok (Predicted Rating: 3.24)
- Ocean Ecopark (Predicted Rating: 3.24)

**Top 5 Recommendations for User:** 48
- Desa Wisata Gamplong (Predicted Rating: 3.89)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.70)
- Tafso Barn (Predicted Rating: 3.68)
- Keraton Surabaya (Predicted Rating: 3.66)
- Lawangwangi Creative Space (Predicted Rating: 3.53)

**Top 5 Recommendations for User:** 49
- Sanghyang Heuleut (Predicted Rating: 3.91)
- Desa Wisata Gamplong (Predicted Rating: 3.90)
- Plaza Indonesia (Predicted Rating: 3.86)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.84)
- Taman Balai Kota Bandung (Predicted Rating: 3.78)

**Top 5 Recommendations for User:** 50
- Taman Keputran (Predicted Rating: 3.84)
- Keraton Surabaya (Predicted Rating: 3.67)
- Gereja Perawan Maria Tak Berdosa Surabaya (Predicted Rating: 3.65)
- Monumen Selamat Datang (Predicted Rating: 3.61)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.57)

**Top 5 Recommendations for User:** 51
- Rumah Sipitung (Predicted Rating: 4.01)
- Pantai Baron (Predicted Rating: 3.98)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.94)
- Museum Mpu Tantular (Predicted Rating: 3.87)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.85)

**Top 5 Recommendations for User:** 52
- Bukit Bintang Yogyakarta (Predicted Rating: 4.13)
- Rumah Sipitung (Predicted Rating: 4.03)
- Ciwangun Indah Camp Official (Predicted Rating: 3.87)
- La Kana Chapel (Predicted Rating: 3.87)
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 3.86)

**Top 5 Recommendations for User:** 53
- Keraton Surabaya (Predicted Rating: 4.07)
- Wisata Agro Edukatif Istana Susu Cibugary (Predicted Rating: 4.05)
- Pantai Baron (Predicted Rating: 3.85)
- Ledok Sambi (Predicted Rating: 3.80)
- Taman Pelangi (Predicted Rating: 3.79)

**Top 5 Recommendations for User:** 54
- Jogja Exotarium (Predicted Rating: 4.01)
- Keraton Surabaya (Predicted Rating: 3.76)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.71)
- Curug Batu Templek (Predicted Rating: 3.66)
- Taman Keputran (Predicted Rating: 3.66)

**Top 5 Recommendations for User:** 55
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.82)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.75)
- Keraton Surabaya (Predicted Rating: 3.74)
- Monumen Jalesveva Jayamahe (Predicted Rating: 3.72)
- Kampung Wisata Taman Sari (Predicted Rating: 3.71)

**Top 5 Recommendations for User:** 56
- Sumur Gumuling (Predicted Rating: 3.70)
- Upside Down World Bandung (Predicted Rating: 3.64)
- Monumen Selamat Datang (Predicted Rating: 3.63)
- Keraton Surabaya (Predicted Rating: 3.59)
- Bukit Jamur (Predicted Rating: 3.59)

**Top 5 Recommendations for User:** 57
- Pasar Kebon Empring Bintaran (Predicted Rating: 3.52)
- Keraton Surabaya (Predicted Rating: 3.51)
- Taman Srigunting (Predicted Rating: 3.45)
- Bukit Moko (Predicted Rating: 3.45)
- Kampung Cina (Predicted Rating: 3.37)

**Top 5 Recommendations for User:** 58
- Monumen Jalesveva Jayamahe (Predicted Rating: 3.79)
- Keraton Surabaya (Predicted Rating: 3.74)
- GPIB Immanuel Semarang (Gereja Blenduk) (Predicted Rating: 3.67)
- Atlantis Land Surabaya (Predicted Rating: 3.67)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.67)

**Top 5 Recommendations for User:** 59
- Keraton Surabaya (Predicted Rating: 4.03)
- Curug Batu Templek (Predicted Rating: 3.95)
- Air Terjun Kedung Pedut (Predicted Rating: 3.91)
- Monumen Nasional (Predicted Rating: 3.91)
- Taman Spathodea (Predicted Rating: 3.85)

**Top 5 Recommendations for User:** 60
- Bukit Jamur (Predicted Rating: 3.90)
- Keraton Surabaya (Predicted Rating: 3.87)
- Monumen Jalesveva Jayamahe (Predicted Rating: 3.74)
- Pintoe Langit Dahromo (Predicted Rating: 3.70)
- Tafso Barn (Predicted Rating: 3.70)

**Top 5 Recommendations for User:** 61
- Pantai Ngrawe (Mesra) (Predicted Rating: 4.32)
- Taman Pelangi (Predicted Rating: 4.27)
- Keraton Surabaya (Predicted Rating: 4.25)
- Gereja Perawan Maria Tak Berdosa Surabaya (Predicted Rating: 4.24)
- Pintoe Langit Dahromo (Predicted Rating: 4.17)

**Top 5 Recommendations for User:** 62
- Taman Keputran (Predicted Rating: 4.13)
- Bukit Bintang Yogyakarta (Predicted Rating: 4.02)
- Tafso Barn (Predicted Rating: 3.99)
- Keraton Surabaya (Predicted Rating: 3.98)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.95)

**Top 5 Recommendations for User:** 63
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.75)
- Grand Maerakaca (Predicted Rating: 3.71)
- Tafso Barn (Predicted Rating: 3.69)
- Keraton Surabaya (Predicted Rating: 3.65)
- Monumen Yogya Kembali (Predicted Rating: 3.58)

**Top 5 Recommendations for User:** 64
- Keraton Surabaya (Predicted Rating: 3.90)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.77)
- Jogja Exotarium (Predicted Rating: 3.76)
- Desa Wisata Lembah Kalipancur (Predicted Rating: 3.73)
- Atlantis Water Adventure (Predicted Rating: 3.72)

**Top 5 Recommendations for User:** 65
- Monumen Selamat Datang (Predicted Rating: 4.11)
- Curug Batu Templek (Predicted Rating: 4.10)
- Keraton Surabaya (Predicted Rating: 3.86)
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 3.85)
- Pantai Baron (Predicted Rating: 3.84)

**Top 5 Recommendations for User:** 66
- Margasatwa Muara Angke (Predicted Rating: 3.61)
- Pemandian Air Panas Ciater (Predicted Rating: 3.51)
- Desa Wisata Gamplong (Predicted Rating: 3.38)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.37)
- Tafso Barn (Predicted Rating: 3.29)

**Top 5 Recommendations for User:** 67
- Desa Wisata Gamplong (Predicted Rating: 3.85)
- Tafso Barn (Predicted Rating: 3.79)
- Pasar Kebon Empring Bintaran (Predicted Rating: 3.71)
- Keraton Surabaya (Predicted Rating: 3.67)
- Taman Balai Kota Bandung (Predicted Rating: 3.62)

**Top 5 Recommendations for User:** 68
- Taman Keputran (Predicted Rating: 4.37)
- Waterpark Kenjeran Surabaya (Predicted Rating: 4.07)
- Tafso Barn (Predicted Rating: 4.05)
- Keraton Surabaya (Predicted Rating: 3.93)
- Glamping Lakeside Rancabali (Predicted Rating: 3.91)

**Top 5 Recommendations for User:** 69
- Curug Tilu Leuwi Opat (Predicted Rating: 3.50)
- Rumah Sipitung (Predicted Rating: 3.46)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.39)
- Keraton Surabaya (Predicted Rating: 3.36)
- Situs Warungboto (Predicted Rating: 3.33)

**Top 5 Recommendations for User:** 70
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.04)
- Sindu Kusuma Edupark (SKE) (Predicted Rating: 3.93)
- Kampung Wisata Taman Sari (Predicted Rating: 3.90)
- Keraton Surabaya (Predicted Rating: 3.90)
- Air Terjun Kedung Pedut (Predicted Rating: 3.90)

**Top 5 Recommendations for User:** 71
- Watu Goyang (Predicted Rating: 3.92)
- Alive Museum Ancol (Predicted Rating: 3.91)
- Monumen Yogya Kembali (Predicted Rating: 3.87)
- Pantai Baron (Predicted Rating: 3.87)
- Hutan Kota Srengseng (Predicted Rating: 3.80)

**Top 5 Recommendations for User:** 72
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.67)
- Bukit Jamur (Predicted Rating: 3.64)
- Desa Wisata Lembah Kalipancur (Predicted Rating: 3.63)
- Watu Goyang (Predicted Rating: 3.60)
- Keraton Surabaya (Predicted Rating: 3.59)

**Top 5 Recommendations for User:** 73
- Keraton Surabaya (Predicted Rating: 3.57)
- Curug Batu Templek (Predicted Rating: 3.50)
- Alun-alun Utara Keraton Yogyakarta (Predicted Rating: 3.48)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.44)
- Air Terjun Kedung Pedut (Predicted Rating: 3.42)

**Top 5 Recommendations for User:** 74
- Monumen Selamat Datang (Predicted Rating: 3.90)
- Monumen Sanapati (Predicted Rating: 3.87)
- Taman Pelangi Yogyakarta (Predicted Rating: 3.87)
- Jogja Exotarium (Predicted Rating: 3.82)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.74)

**Top 5 Recommendations for User:** 75
- Keraton Surabaya (Predicted Rating: 4.04)
- Bukit Bintang Yogyakarta (Predicted Rating: 4.03)
- Monumen Yogya Kembali (Predicted Rating: 4.02)
- Pantai Baron (Predicted Rating: 4.00)
- Wisata Lereng Kelir (Predicted Rating: 3.89)

**Top 5 Recommendations for User:** 76
- Keraton Surabaya (Predicted Rating: 3.74)
- Museum Gunung Merapi (Predicted Rating: 3.53)
- Glamping Lakeside Rancabali (Predicted Rating: 3.45)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.40)
- Air Terjun Kedung Pedut (Predicted Rating: 3.39)

**Top 5 Recommendations for User:** 77
- Bukit Jamur (Predicted Rating: 3.91)
- Tafso Barn (Predicted Rating: 3.88)
- Keraton Surabaya (Predicted Rating: 3.80)
- Kampung Wisata Taman Sari (Predicted Rating: 3.74)
- Desa Wisata Gamplong (Predicted Rating: 3.73)

**Top 5 Recommendations for User:** 78
- Keraton Surabaya (Predicted Rating: 4.28)
- Monumen Nasional (Predicted Rating: 4.10)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 4.09)
- Wisata Alam Mangrove Angke (Predicted Rating: 4.07)
- Water Park Bandung Indah (Predicted Rating: 4.01)

**Top 5 Recommendations for User:** 79
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.80)
- Teras Cikapundung BBWS (Predicted Rating: 3.72)
- Pantai Baron (Predicted Rating: 3.68)
- Mountain View Golf Club (Predicted Rating: 3.48)
- Margasatwa Muara Angke (Predicted Rating: 3.48)

**Top 5 Recommendations for User:** 80
- Taman Hiburan Rakyat (Predicted Rating: 3.55)
- Pantai Congot (Predicted Rating: 3.45)
- Keraton Surabaya (Predicted Rating: 3.32)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.31)
- Pintoe Langit Dahromo (Predicted Rating: 3.30)

**Top 5 Recommendations for User:** 81
- Keraton Surabaya (Predicted Rating: 3.53)
- Monumen Nasional (Predicted Rating: 3.35)
- Desa Wisata Gamplong (Predicted Rating: 3.34)
- Monumen Yogya Kembali (Predicted Rating: 3.32)
- Bukit Jamur (Predicted Rating: 3.29)

**Top 5 Recommendations for User:** 82
- Obyek Wisata Goa Kreo (Predicted Rating: 3.90)
- Food Junction Grand Pakuwon (Predicted Rating: 3.90)
- Rainbow Garden (Predicted Rating: 3.87)
- Air Terjun Kedung Pedut (Predicted Rating: 3.84)
- Kampung Cina (Predicted Rating: 3.83)

**Top 5 Recommendations for User:** 83
- Tafso Barn (Predicted Rating: 3.92)
- Pantai Baron (Predicted Rating: 3.92)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.83)
- Dago Dreampark (Predicted Rating: 3.80)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.78)

**Top 5 Recommendations for User:** 84
- Pintoe Langit Dahromo (Predicted Rating: 4.19)
- Sumur Gumuling (Predicted Rating: 4.19)
- Keraton Surabaya (Predicted Rating: 4.13)
- Taman Srigunting (Predicted Rating: 4.07)
- Air Terjun Kedung Pedut (Predicted Rating: 4.07)

**Top 5 Recommendations for User:** 85
- Keraton Surabaya (Predicted Rating: 3.73)
- Pantai Congot (Predicted Rating: 3.71)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.47)
- Bukit Jamur (Predicted Rating: 3.43)
- Sumur Gumuling (Predicted Rating: 3.42)

**Top 5 Recommendations for User:** 86
- Monumen Selamat Datang (Predicted Rating: 4.03)
- Taman Spathodea (Predicted Rating: 3.88)
- Keraton Surabaya (Predicted Rating: 3.80)
- Rumah Sipitung (Predicted Rating: 3.80)
- Bukit Jamur (Predicted Rating: 3.75)

**Top 5 Recommendations for User:** 87
- Sanghyang Heuleut (Predicted Rating: 3.97)
- Monumen Yogya Kembali (Predicted Rating: 3.89)
- Rainbow Garden (Predicted Rating: 3.81)
- Keraton Surabaya (Predicted Rating: 3.78)
- Kampung Cina (Predicted Rating: 3.73)

**Top 5 Recommendations for User:** 88
- Keraton Surabaya (Predicted Rating: 3.16)
- Wisata Kraton Jogja (Predicted Rating: 2.93)
- Teras Cikapundung BBWS (Predicted Rating: 2.91)
- Kampung Cina (Predicted Rating: 2.91)
- Pantai Baron (Predicted Rating: 2.90)

**Top 5 Recommendations for User:** 89
- Kampung Cina (Predicted Rating: 3.84)
- Air Terjun Kedung Pedut (Predicted Rating: 3.83)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.81)
- Keraton Surabaya (Predicted Rating: 3.74)
- Museum Mpu Tantular (Predicted Rating: 3.74)

**Top 5 Recommendations for User:** 90
- Keraton Surabaya (Predicted Rating: 3.47)
- Monumen Selamat Datang (Predicted Rating: 3.29)
- Sumur Gumuling (Predicted Rating: 3.29)
- Curug Batu Templek (Predicted Rating: 3.26)
- Taman Pelangi Yogyakarta (Predicted Rating: 3.25)

**Top 5 Recommendations for User:** 91
- Bukit Wisata Pulepayung (Predicted Rating: 3.82)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.80)
- Bukit Jamur (Predicted Rating: 3.80)
- Pantai Congot (Predicted Rating: 3.73)
- Taman Hiburan Rakyat (Predicted Rating: 3.72)

**Top 5 Recommendations for User:** 92
- Keraton Surabaya (Predicted Rating: 4.37)
- Desa Wisata Gamplong (Predicted Rating: 4.15)
- Observatorium Bosscha (Predicted Rating: 4.12)
- Monumen Yogya Kembali (Predicted Rating: 4.11)
- Monumen Nasional (Predicted Rating: 4.11)

**Top 5 Recommendations for User:** 93
- Gereja Perawan Maria Tak Berdosa Surabaya (Predicted Rating: 4.17)
- Taman Hiburan Rakyat (Predicted Rating: 4.09)
- Keraton Surabaya (Predicted Rating: 4.07)
- Taman Keputran (Predicted Rating: 4.00)
- Monumen Yogya Kembali (Predicted Rating: 3.98)

**Top 5 Recommendations for User:** 94
- Keraton Surabaya (Predicted Rating: 3.96)
- Monumen Sanapati (Predicted Rating: 3.80)
- Selasar Sunaryo Art Space (Predicted Rating: 3.79)
- Monumen Yogya Kembali (Predicted Rating: 3.72)
- Wisata Agro Edukatif Istana Susu Cibugary (Predicted Rating: 3.68)

**Top 5 Recommendations for User:** 95
- Keraton Surabaya (Predicted Rating: 4.36)
- Monumen Yogya Kembali (Predicted Rating: 4.12)
- Pasar Kebon Empring Bintaran (Predicted Rating: 4.11)
- Ciwangun Indah Camp Official (Predicted Rating: 4.05)
- Pantai Baron (Predicted Rating: 3.99)

**Top 5 Recommendations for User:** 96
- Keraton Surabaya (Predicted Rating: 3.74)
- Teras Cikapundung BBWS (Predicted Rating: 3.60)
- Bukit Jamur (Predicted Rating: 3.59)
- Stone Garden Citatah (Predicted Rating: 3.50)
- Ledok Sambi (Predicted Rating: 3.50)

**Top 5 Recommendations for User:** 97
- Keraton Surabaya (Predicted Rating: 4.02)
- Curug Batu Templek (Predicted Rating: 4.01)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.85)
- Atlantis Water Adventure (Predicted Rating: 3.79)
- Bukit Jamur (Predicted Rating: 3.79)

**Top 5 Recommendations for User:** 98
- Pantai Ngrenehan (Predicted Rating: 3.77)
- Goa Pindul (Predicted Rating: 3.76)
- Monumen Kapal Selam (Predicted Rating: 3.76)
- Kampung Cina (Predicted Rating: 3.76)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.75)

**Top 5 Recommendations for User:** 99
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.15)
- Keraton Surabaya (Predicted Rating: 4.07)
- Rumah Sipitung (Predicted Rating: 4.04)
- La Kana Chapel (Predicted Rating: 4.02)
- Tafso Barn (Predicted Rating: 3.97)

**Top 5 Recommendations for User:** 100
- Pantai Baron (Predicted Rating: 3.78)
- Waterpark Kenjeran Surabaya (Predicted Rating: 3.77)
- Tafso Barn (Predicted Rating: 3.68)
- Pantai Kesirat (Predicted Rating: 3.58)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.56)

**Top 5 Recommendations for User:** 101
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.84)
- Skyrink - Mall Taman Anggrek (Predicted Rating: 3.80)
- Keraton Surabaya (Predicted Rating: 3.75)
- Dago Dreampark (Predicted Rating: 3.74)
- Alive Museum Ancol (Predicted Rating: 3.73)

**Top 5 Recommendations for User:** 102
- Kampung Wisata Taman Sari (Predicted Rating: 4.03)
- Setu Babakan (Predicted Rating: 3.54)
- Bukit Jamur (Predicted Rating: 3.49)
- Kota Mini (Predicted Rating: 3.47)
- Monumen Yogya Kembali (Predicted Rating: 3.45)

**Top 5 Recommendations for User:** 103
- Keraton Surabaya (Predicted Rating: 3.83)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.80)
- Taman Kasmaran (Predicted Rating: 3.74)
- Watu Goyang (Predicted Rating: 3.64)
- Kampung Cina (Predicted Rating: 3.63)

**Top 5 Recommendations for User:** 104
- Keraton Surabaya (Predicted Rating: 4.03)
- Jogja Exotarium (Predicted Rating: 3.93)
- Pemandian Air Panas Ciater (Predicted Rating: 3.88)
- Glamping Lakeside Rancabali (Predicted Rating: 3.81)
- Teras Cikapundung BBWS (Predicted Rating: 3.74)

**Top 5 Recommendations for User:** 105
- Stone Garden Citatah (Predicted Rating: 4.39)
- Keraton Surabaya (Predicted Rating: 4.38)
- Bukit Jamur (Predicted Rating: 4.27)
- Air Terjun Kedung Pedut (Predicted Rating: 4.26)
- Monumen Nasional (Predicted Rating: 4.20)

**Top 5 Recommendations for User:** 106
- Watu Goyang (Predicted Rating: 3.97)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.83)
- Pantai Baron (Predicted Rating: 3.76)
- Monumen Yogya Kembali (Predicted Rating: 3.75)
- Keraton Surabaya (Predicted Rating: 3.71)

**Top 5 Recommendations for User:** 107
- Monumen Selamat Datang (Predicted Rating: 3.83)
- Seribu Batu Songgo Langit (Predicted Rating: 3.77)
- Bukit Jamur (Predicted Rating: 3.75)
- Masjid Nasional Al-Akbar (Predicted Rating: 3.72)
- Desa Wisata Gamplong (Predicted Rating: 3.70)

**Top 5 Recommendations for User:** 108
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.17)
- Keraton Surabaya (Predicted Rating: 4.07)
- Air Terjun Kedung Pedut (Predicted Rating: 4.03)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.96)
- Pemandian Air Panas Ciater (Predicted Rating: 3.92)

**Top 5 Recommendations for User:** 109
- Goa Pindul (Predicted Rating: 4.09)
- Grand Maerakaca (Predicted Rating: 4.02)
- Monumen Sanapati (Predicted Rating: 4.01)
- Atlantis Water Adventure (Predicted Rating: 4.00)
- Dago Dreampark (Predicted Rating: 3.99)

**Top 5 Recommendations for User:** 110
- Bukit Jamur (Predicted Rating: 3.88)
- Trans Studio Bandung (Predicted Rating: 3.66)
- Keraton Surabaya (Predicted Rating: 3.62)
- Taman Spathodea (Predicted Rating: 3.61)
- Monumen Nasional (Predicted Rating: 3.58)

**Top 5 Recommendations for User:** 111
- Sanghyang Heuleut (Predicted Rating: 4.18)
- Teras Cikapundung BBWS (Predicted Rating: 4.15)
- Pantai Parangtritis (Predicted Rating: 4.10)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.02)
- Kampung Cina (Predicted Rating: 3.99)

**Top 5 Recommendations for User:** 112
- Old City 3D Trick Art Museum (Predicted Rating: 3.85)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.80)
- Jogja Exotarium (Predicted Rating: 3.78)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.71)
- Keraton Surabaya (Predicted Rating: 3.70)

**Top 5 Recommendations for User:** 113
- Trans Studio Bandung (Predicted Rating: 4.09)
- Puspa Iptek Sundial (Predicted Rating: 4.03)
- Monumen Nasional (Predicted Rating: 4.00)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.97)
- Keraton Surabaya (Predicted Rating: 3.94)

**Top 5 Recommendations for User:** 114
- Keraton Surabaya (Predicted Rating: 3.85)
- Curug Batu Templek (Predicted Rating: 3.73)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.72)
- Situs Warungboto (Predicted Rating: 3.72)
- Glamping Lakeside Rancabali (Predicted Rating: 3.72)

**Top 5 Recommendations for User:** 115
- Monumen Nasional (Predicted Rating: 4.28)
- Museum Bank Indonesia (Predicted Rating: 4.28)
- Taman Hiburan Rakyat (Predicted Rating: 4.27)
- Bukit Jamur (Predicted Rating: 4.19)
- Selasar Sunaryo Art Space (Predicted Rating: 4.17)

**Top 5 Recommendations for User:** 116
- Teras Cikapundung BBWS (Predicted Rating: 3.60)
- Ekowisata Mangrove Wonorejo (Predicted Rating: 3.49)
- Pantai Baron (Predicted Rating: 3.42)
- Keraton Surabaya (Predicted Rating: 3.41)
- Food Junction Grand Pakuwon (Predicted Rating: 3.41)

**Top 5 Recommendations for User:** 117
- Rainbow Garden (Predicted Rating: 4.26)
- Ciwangun Indah Camp Official (Predicted Rating: 4.04)
- Keraton Surabaya (Predicted Rating: 4.03)
- Teras Cikapundung BBWS (Predicted Rating: 3.85)
- Pantai Baron (Predicted Rating: 3.81)

**Top 5 Recommendations for User:** 118
- Klenteng Jin De Yuan (Predicted Rating: 4.04)
- Bukit Jamur (Predicted Rating: 4.01)
- Keraton Surabaya (Predicted Rating: 3.91)
- Pantai Baron (Predicted Rating: 3.85)
- Monumen Nasional (Predicted Rating: 3.84)

**Top 5 Recommendations for User:** 119
- Keraton Surabaya (Predicted Rating: 3.75)
- Glamping Lakeside Rancabali (Predicted Rating: 3.59)
- Air Terjun Kedung Pedut (Predicted Rating: 3.53)
- Taman Srigunting (Predicted Rating: 3.50)
- Jogja Bay Pirates Adventure Waterpark (Predicted Rating: 3.48)

**Top 5 Recommendations for User:** 120
- Keraton Surabaya (Predicted Rating: 4.14)
- Wisata Agro Edukatif Istana Susu Cibugary (Predicted Rating: 3.92)
- Pantai Baron (Predicted Rating: 3.89)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.88)
- Pantai Ngrenehan (Predicted Rating: 3.87)

**Top 5 Recommendations for User:** 121
- Pantai Baron (Predicted Rating: 3.94)
- Monumen Nasional (Predicted Rating: 3.75)
- Bukit Jamur (Predicted Rating: 3.71)
- Kampung Wisata Taman Sari (Predicted Rating: 3.70)
- Alive Museum Ancol (Predicted Rating: 3.66)

**Top 5 Recommendations for User:** 122
- Keraton Surabaya (Predicted Rating: 3.96)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.87)
- Teras Cikapundung BBWS (Predicted Rating: 3.82)
- Jurang Tembelan Kanigoro (Predicted Rating: 3.82)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.73)

**Top 5 Recommendations for User:** 123
- Keraton Surabaya (Predicted Rating: 3.11)
- Taman Pelangi (Predicted Rating: 3.07)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.04)
- Pantai Baron (Predicted Rating: 3.01)
- Bukit Bintang Yogyakarta (Predicted Rating: 2.99)

**Top 5 Recommendations for User:** 124
- Kawasan Malioboro (Predicted Rating: 3.82)
- Goa Pindul (Predicted Rating: 3.66)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.66)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.65)
- Tafso Barn (Predicted Rating: 3.62)

**Top 5 Recommendations for User:** 125
- Candi Ijo (Predicted Rating: 4.08)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.81)
- Tugu Muda Semarang (Predicted Rating: 3.80)
- Rainbow Garden (Predicted Rating: 3.76)
- Ekowisata Mangrove Wonorejo (Predicted Rating: 3.75)

**Top 5 Recommendations for User:** 126
- Keraton Surabaya (Predicted Rating: 4.32)
- Bukit Jamur (Predicted Rating: 4.30)
- Taman Hiburan Rakyat (Predicted Rating: 4.23)
- Bukit Bintang Yogyakarta (Predicted Rating: 4.06)
- Taman Pelangi Yogyakarta (Predicted Rating: 4.06)

**Top 5 Recommendations for User:** 127
- Kampung Wisata Taman Sari (Predicted Rating: 3.85)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.85)
- Pantai Baron (Predicted Rating: 3.70)
- Keraton Surabaya (Predicted Rating: 3.69)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.67)

**Top 5 Recommendations for User:** 128
- Taman Pelangi (Predicted Rating: 3.84)
- Dago Dreampark (Predicted Rating: 3.81)
- Embung Tambakboyo (Predicted Rating: 3.77)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.74)
- Monumen Yogya Kembali (Predicted Rating: 3.70)

**Top 5 Recommendations for User:** 129
- Museum Nike Ardilla (Predicted Rating: 4.25)
- Keraton Surabaya (Predicted Rating: 4.25)
- Curug Tilu Leuwi Opat (Predicted Rating: 4.09)
- Monumen Yogya Kembali (Predicted Rating: 4.04)
- Rumah Sipitung (Predicted Rating: 4.03)

**Top 5 Recommendations for User:** 130
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 4.41)
- Bukit Bintang Yogyakarta (Predicted Rating: 4.31)
- Bukit Jamur (Predicted Rating: 4.27)
- Keraton Surabaya (Predicted Rating: 4.24)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.19)

**Top 5 Recommendations for User:** 131
- Obyek Wisata Goa Kreo (Predicted Rating: 3.88)
- Kampung Wisata Kadipaten (Predicted Rating: 3.70)
- Pantai Baron (Predicted Rating: 3.69)
- Keraton Surabaya (Predicted Rating: 3.61)
- Kampung Cina (Predicted Rating: 3.52)

**Top 5 Recommendations for User:** 132
- Wisata Lereng Kelir (Predicted Rating: 3.81)
- Pantai Kesirat (Predicted Rating: 3.70)
- Tafso Barn (Predicted Rating: 3.69)
- Keraton Surabaya (Predicted Rating: 3.68)
- Obyek Wisata Goa Kreo (Predicted Rating: 3.67)

**Top 5 Recommendations for User:** 133
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.15)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.75)
- Pantai Drini (Predicted Rating: 3.74)
- Teras Cikapundung BBWS (Predicted Rating: 3.71)
- Glamping Lakeside Rancabali (Predicted Rating: 3.70)

**Top 5 Recommendations for User:** 134
- Pantai Baron (Predicted Rating: 4.10)
- Gumuk Pasir Parangkusumo (Predicted Rating: 3.91)
- Keraton Surabaya (Predicted Rating: 3.91)
- Skyrink - Mall Taman Anggrek (Predicted Rating: 3.79)
- Glamping Lakeside Rancabali (Predicted Rating: 3.77)

**Top 5 Recommendations for User:** 135
- Pantai Baron (Predicted Rating: 3.81)
- Air Terjun Kedung Pedut (Predicted Rating: 3.80)
- Keraton Surabaya (Predicted Rating: 3.79)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.73)
- Kampung Cina (Predicted Rating: 3.71)

**Top 5 Recommendations for User:** 136
- Sanghyang Heuleut (Predicted Rating: 3.28)
- Jogja Exotarium (Predicted Rating: 3.28)
- Monumen Selamat Datang (Predicted Rating: 3.25)
- Tafso Barn (Predicted Rating: 3.22)
- Margasatwa Muara Angke (Predicted Rating: 3.21)

**Top 5 Recommendations for User:** 137
- Keraton Surabaya (Predicted Rating: 4.55)
- Glamping Lakeside Rancabali (Predicted Rating: 4.21)
- Kampoeng Tulip (Predicted Rating: 4.03)
- Situs Warungboto (Predicted Rating: 4.01)
- Taman Spathodea (Predicted Rating: 4.01)

**Top 5 Recommendations for User:** 138
- Jogja Exotarium (Predicted Rating: 4.01)
- Keraton Surabaya (Predicted Rating: 3.90)
- Monumen Sanapati (Predicted Rating: 3.88)
- Taman Spathodea (Predicted Rating: 3.85)
- Curug Luhur Waterfall (Predicted Rating: 3.76)

**Top 5 Recommendations for User:** 139
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.20)
- Goa Pindul (Predicted Rating: 4.14)
- Dago Dreampark (Predicted Rating: 3.97)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.90)
- Curug Batu Templek (Predicted Rating: 3.83)

**Top 5 Recommendations for User:** 140
- Bukit Jamur (Predicted Rating: 4.19)
- Keraton Surabaya (Predicted Rating: 4.06)
- Monumen Nasional (Predicted Rating: 4.05)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.96)
- Dago Dreampark (Predicted Rating: 3.96)

**Top 5 Recommendations for User:** 141
- Curug Batu Templek (Predicted Rating: 4.08)
- Desa Wisata Sungai Code Jogja Kota (Predicted Rating: 4.02)
- Atlantis Water Adventure (Predicted Rating: 3.96)
- Keraton Surabaya (Predicted Rating: 3.95)
- Pantai Baron (Predicted Rating: 3.93)

**Top 5 Recommendations for User:** 142
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.68)
- Pantai Baron (Predicted Rating: 3.54)
- Keraton Surabaya (Predicted Rating: 3.52)
- Taman Flora Bratang Surabaya (Predicted Rating: 3.42)
- Kampung Cina (Predicted Rating: 3.39)

**Top 5 Recommendations for User:** 143
- Air Terjun Kedung Pedut (Predicted Rating: 4.39)
- Tafso Barn (Predicted Rating: 4.20)
- Pantai Wediombo (Predicted Rating: 4.12)
- Old City 3D Trick Art Museum (Predicted Rating: 4.08)
- Embung Tambakboyo (Predicted Rating: 4.00)

**Top 5 Recommendations for User:** 144
- Keraton Surabaya (Predicted Rating: 4.08)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.98)
- Istana Negara Republik Indonesia (Predicted Rating: 3.94)
- Surabaya Museum (Gedung Siola) (Predicted Rating: 3.88)
- Kampung Cina (Predicted Rating: 3.87)

**Top 5 Recommendations for User:** 145
- Keraton Surabaya (Predicted Rating: 3.94)
- Sumur Gumuling (Predicted Rating: 3.85)
- Monumen Yogya Kembali (Predicted Rating: 3.82)
- Ocean Ecopark (Predicted Rating: 3.77)
- Dago Dreampark (Predicted Rating: 3.69)

**Top 5 Recommendations for User:** 146
- Keraton Surabaya (Predicted Rating: 3.72)
- Watu Goyang (Predicted Rating: 3.53)
- Monumen Selamat Datang (Predicted Rating: 3.45)
- Curug Batu Templek (Predicted Rating: 3.40)
- Dago Dreampark (Predicted Rating: 3.37)

**Top 5 Recommendations for User:** 147
- Desa Wisata Sungai Code Jogja Kota (Predicted Rating: 3.77)
- Bukit Jamur (Predicted Rating: 3.76)
- Keraton Surabaya (Predicted Rating: 3.76)
- Pantai Baron (Predicted Rating: 3.74)
- Kampung Wisata Taman Sari (Predicted Rating: 3.72)

**Top 5 Recommendations for User:** 148
- NuArt Sculpture Park (Predicted Rating: 3.84)
- Air Terjun Kedung Pedut (Predicted Rating: 3.78)
- Jogja Exotarium (Predicted Rating: 3.74)
- Keraton Surabaya (Predicted Rating: 3.69)
- Kampung Cina (Predicted Rating: 3.68)

**Top 5 Recommendations for User:** 149
- Pantai Baron (Predicted Rating: 4.17)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.73)
- Alun-alun Utara Keraton Yogyakarta (Predicted Rating: 3.70)
- Obyek Wisata Goa Kreo (Predicted Rating: 3.67)
- Taman Hiburan Rakyat (Predicted Rating: 3.66)

**Top 5 Recommendations for User:** 150
- Bukit Jamur (Predicted Rating: 4.25)
- Alive Museum Ancol (Predicted Rating: 4.07)
- Monumen Nasional (Predicted Rating: 4.05)
- Desa Wisata Gamplong (Predicted Rating: 4.04)
- Stone Garden Citatah (Predicted Rating: 4.00)

**Top 5 Recommendations for User:** 151
- Goa Pindul (Predicted Rating: 4.32)
- Keraton Surabaya (Predicted Rating: 4.03)
- Ciwangun Indah Camp Official (Predicted Rating: 3.99)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.96)
- Museum Nike Ardilla (Predicted Rating: 3.96)

**Top 5 Recommendations for User:** 152
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 3.95)
- Kampung Cina (Predicted Rating: 3.87)
- Keraton Surabaya (Predicted Rating: 3.77)
- Teras Cikapundung BBWS (Predicted Rating: 3.70)
- Pantai Baron (Predicted Rating: 3.69)

**Top 5 Recommendations for User:** 153
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.80)
- Jogja Bay Pirates Adventure Waterpark (Predicted Rating: 3.73)
- Taman Keputran (Predicted Rating: 3.43)
- Grand Maerakaca (Predicted Rating: 3.43)
- Monumen Yogya Kembali (Predicted Rating: 3.38)

**Top 5 Recommendations for User:** 154
- Keraton Surabaya (Predicted Rating: 4.37)
- Pantai Baron (Predicted Rating: 4.12)
- Sumur Gumuling (Predicted Rating: 4.07)
- Bukit Bintang Yogyakarta (Predicted Rating: 4.01)
- Pantai Congot (Predicted Rating: 4.01)

**Top 5 Recommendations for User:** 155
- Rainbow Garden (Predicted Rating: 3.75)
- Glamping Lakeside Rancabali (Predicted Rating: 3.71)
- Pantai Baron (Predicted Rating: 3.54)
- Keraton Surabaya (Predicted Rating: 3.50)
- Obyek Wisata Goa Kreo (Predicted Rating: 3.48)

**Top 5 Recommendations for User:** 156
- Bukit Wisata Pulepayung (Predicted Rating: 4.27)
- Pantai Greweng (Predicted Rating: 4.06)
- Ocean Ecopark (Predicted Rating: 3.98)
- Teras Cikapundung BBWS (Predicted Rating: 3.97)
- Selasar Sunaryo Art Space (Predicted Rating: 3.97)

**Top 5 Recommendations for User:** 157
- Keraton Surabaya (Predicted Rating: 4.32)
- Situs Warungboto (Predicted Rating: 3.99)
- Pantai Baron (Predicted Rating: 3.97)
- Trans Studio Bandung (Predicted Rating: 3.97)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.91)

**Top 5 Recommendations for User:** 158
- Kampung Cina (Predicted Rating: 3.94)
- Alive Museum Ancol (Predicted Rating: 3.81)
- Atlantis Land Surabaya (Predicted Rating: 3.78)
- Grojogan Watu Purbo Bangunrejo (Predicted Rating: 3.75)
- Monumen Jalesveva Jayamahe (Predicted Rating: 3.74)

**Top 5 Recommendations for User:** 159
- Bukit Jamur (Predicted Rating: 3.73)
- Jogja Exotarium (Predicted Rating: 3.62)
- Keraton Surabaya (Predicted Rating: 3.56)
- Pantai Indrayanti (Predicted Rating: 3.54)
- Taman Spathodea (Predicted Rating: 3.53)

**Top 5 Recommendations for User:** 160
- Air Terjun Kedung Pedut (Predicted Rating: 4.05)
- Jurang Tembelan Kanigoro (Predicted Rating: 3.91)
- Keraton Surabaya (Predicted Rating: 3.86)
- GunungTangkuban perahu (Predicted Rating: 3.66)
- Kampung Cina (Predicted Rating: 3.63)

**Top 5 Recommendations for User:** 161
- Kampung Cina (Predicted Rating: 4.01)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.96)
- Keraton Surabaya (Predicted Rating: 3.93)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.83)
- Bukit Wisata Pulepayung (Predicted Rating: 3.80)

**Top 5 Recommendations for User:** 162
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.14)
- Monumen Nasional (Predicted Rating: 4.05)
- Jogja Exotarium (Predicted Rating: 4.03)
- Situs Warungboto (Predicted Rating: 4.03)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.00)

**Top 5 Recommendations for User:** 163
- Bukit Jamur (Predicted Rating: 4.20)
- Sumur Gumuling (Predicted Rating: 3.94)
- Saung Angklung Mang Udjo (Predicted Rating: 3.91)
- Pantai Baron (Predicted Rating: 3.91)
- Teras Cikapundung BBWS (Predicted Rating: 3.84)

**Top 5 Recommendations for User:** 164
- Bukit Jamur (Predicted Rating: 3.91)
- Kampoeng Kopi Banaran (Predicted Rating: 3.89)
- Keraton Surabaya (Predicted Rating: 3.87)
- Desa Wisata Gamplong (Predicted Rating: 3.86)
- Tafso Barn (Predicted Rating: 3.84)

**Top 5 Recommendations for User:** 165
- Kampung Cina (Predicted Rating: 4.07)
- Keraton Surabaya (Predicted Rating: 4.03)
- Pantai Baron (Predicted Rating: 4.01)
- Sumur Gumuling (Predicted Rating: 3.99)
- Monumen Yogya Kembali (Predicted Rating: 3.94)

**Top 5 Recommendations for User:** 166
- Keraton Surabaya (Predicted Rating: 3.97)
- Bukit Jamur (Predicted Rating: 3.92)
- Taman Keputran (Predicted Rating: 3.86)
- Kota Mini (Predicted Rating: 3.85)
- Pasar Kebon Empring Bintaran (Predicted Rating: 3.84)

**Top 5 Recommendations for User:** 167
- Kota Mini (Predicted Rating: 4.00)
- Monumen Nasional (Predicted Rating: 3.98)
- Sumur Gumuling (Predicted Rating: 3.97)
- Kampung Cina (Predicted Rating: 3.93)
- Kampung Wisata Taman Sari (Predicted Rating: 3.93)

**Top 5 Recommendations for User:** 168
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.14)
- Keraton Surabaya (Predicted Rating: 4.10)
- Bukit Bintang Yogyakarta (Predicted Rating: 4.09)
- Glamping Lakeside Rancabali (Predicted Rating: 4.06)
- Bukit Jamur (Predicted Rating: 4.06)

**Top 5 Recommendations for User:** 169
- Rainbow Garden (Predicted Rating: 4.24)
- Monumen Nasional (Predicted Rating: 4.19)
- Situs Warungboto (Predicted Rating: 4.12)
- Keraton Surabaya (Predicted Rating: 4.00)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.95)

**Top 5 Recommendations for User:** 170
- Stone Garden Citatah (Predicted Rating: 3.80)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.47)
- Kampung Cina (Predicted Rating: 3.47)
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 3.44)
- Pasar Kebon Empring Bintaran (Predicted Rating: 3.42)

**Top 5 Recommendations for User:** 171
- Monumen Sanapati (Predicted Rating: 3.86)
- Keraton Surabaya (Predicted Rating: 3.71)
- Air Terjun Kedung Pedut (Predicted Rating: 3.70)
- Curug Batu Templek (Predicted Rating: 3.69)
- Kampung Cina (Predicted Rating: 3.62)

**Top 5 Recommendations for User:** 172
- Bukit Bintang Yogyakarta (Predicted Rating: 3.91)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.88)
- Keraton Surabaya (Predicted Rating: 3.82)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.79)
- Pantai Baron (Predicted Rating: 3.79)

**Top 5 Recommendations for User:** 173
- Keraton Surabaya (Predicted Rating: 3.57)
- Bukit Jamur (Predicted Rating: 3.54)
- Kota Mini (Predicted Rating: 3.53)
- Pantai Baron (Predicted Rating: 3.51)
- Dago Dreampark (Predicted Rating: 3.49)

**Top 5 Recommendations for User:** 174
- Keraton Surabaya (Predicted Rating: 3.79)
- Bukit Jamur (Predicted Rating: 3.71)
- Pantai Indrayanti (Predicted Rating: 3.67)
- Monumen Yogya Kembali (Predicted Rating: 3.61)
- Kota Mini (Predicted Rating: 3.61)

**Top 5 Recommendations for User:** 175
- Rainbow Garden (Predicted Rating: 4.08)
- Kampung Cina (Predicted Rating: 4.06)
- Ciwangun Indah Camp Official (Predicted Rating: 4.01)
- Keraton Surabaya (Predicted Rating: 3.94)
- Teras Cikapundung BBWS (Predicted Rating: 3.93)

**Top 5 Recommendations for User:** 176
- Keraton Surabaya (Predicted Rating: 4.07)
- Taman Budaya Yogyakarta (Predicted Rating: 3.93)
- Air Terjun Kedung Pedut (Predicted Rating: 3.89)
- Monumen Nasional (Predicted Rating: 3.88)
- Monumen Kapal Selam (Predicted Rating: 3.88)

**Top 5 Recommendations for User:** 177
- Keraton Surabaya (Predicted Rating: 4.14)
- Tafso Barn (Predicted Rating: 4.12)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.04)
- Monumen Yogya Kembali (Predicted Rating: 4.04)
- Teras Cikapundung BBWS (Predicted Rating: 3.99)

**Top 5 Recommendations for User:** 178
- Keraton Surabaya (Predicted Rating: 4.00)
- Rumah Sipitung (Predicted Rating: 3.75)
- Glamping Lakeside Rancabali (Predicted Rating: 3.71)
- Pantai Baron (Predicted Rating: 3.69)
- Curug Tilu Leuwi Opat (Predicted Rating: 3.64)

**Top 5 Recommendations for User:** 179
- Bukit Jamur (Predicted Rating: 4.02)
- Keraton Surabaya (Predicted Rating: 3.92)
- Taman Lansia (Predicted Rating: 3.90)
- Taman Keputran (Predicted Rating: 3.83)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.83)

**Top 5 Recommendations for User:** 180
- Jogja Exotarium (Predicted Rating: 3.52)
- Glamping Lakeside Rancabali (Predicted Rating: 3.39)
- Ciwangun Indah Camp Official (Predicted Rating: 3.38)
- Kampung Cina (Predicted Rating: 3.38)
- Curug Batu Templek (Predicted Rating: 3.37)

**Top 5 Recommendations for User:** 181
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 3.65)
- Selasar Sunaryo Art Space (Predicted Rating: 3.63)
- Observatorium Bosscha (Predicted Rating: 3.61)
- Desa Wisata Gamplong (Predicted Rating: 3.60)
- Ocean Ecopark (Predicted Rating: 3.59)

**Top 5 Recommendations for User:** 182
- Bukit Jamur (Predicted Rating: 3.85)
- Pantai Baron (Predicted Rating: 3.80)
- Wisata Lereng Kelir (Predicted Rating: 3.71)
- Tafso Barn (Predicted Rating: 3.71)
- Keraton Surabaya (Predicted Rating: 3.64)

**Top 5 Recommendations for User:** 183
- Bukit Jamur (Predicted Rating: 3.53)
- Seribu Batu Songgo Langit (Predicted Rating: 3.51)
- Pantai Baron (Predicted Rating: 3.50)
- Pasar Kebon Empring Bintaran (Predicted Rating: 3.48)
- Keraton Surabaya (Predicted Rating: 3.47)

**Top 5 Recommendations for User:** 184
- Keraton Surabaya (Predicted Rating: 3.71)
- Dago Dreampark (Predicted Rating: 3.65)
- Skyrink - Mall Taman Anggrek (Predicted Rating: 3.59)
- Air Terjun Kedung Pedut (Predicted Rating: 3.57)
- Bukit Jamur (Predicted Rating: 3.56)

**Top 5 Recommendations for User:** 185
- Monumen Yogya Kembali (Predicted Rating: 3.77)
- Pantai Baron (Predicted Rating: 3.75)
- Wisata Agro Edukatif Istana Susu Cibugary (Predicted Rating: 3.74)
- Taman Keputran (Predicted Rating: 3.67)
- Margasatwa Muara Angke (Predicted Rating: 3.66)

**Top 5 Recommendations for User:** 186
- Taman Hiburan Rakyat (Predicted Rating: 4.04)
- Monumen Yogya Kembali (Predicted Rating: 4.02)
- Bukit Wisata Pulepayung (Predicted Rating: 3.96)
- Taman Srigunting (Predicted Rating: 3.88)
- Tafso Barn (Predicted Rating: 3.83)

**Top 5 Recommendations for User:** 187
- Monumen Selamat Datang (Predicted Rating: 3.83)
- Taman Hiburan Rakyat (Predicted Rating: 3.77)
- Glamping Lakeside Rancabali (Predicted Rating: 3.77)
- Keraton Surabaya (Predicted Rating: 3.75)
- Curug Batu Templek (Predicted Rating: 3.72)

**Top 5 Recommendations for User:** 188
- Pasar Kebon Empring Bintaran (Predicted Rating: 3.35)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.35)
- Bukit Moko (Predicted Rating: 3.26)
- Monumen Jalesveva Jayamahe (Predicted Rating: 3.26)
- Sanghyang Heuleut (Predicted Rating: 3.25)

**Top 5 Recommendations for User:** 189
- Keraton Surabaya (Predicted Rating: 4.33)
- Monumen Nasional (Predicted Rating: 4.26)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 4.18)
- Bukit Jamur (Predicted Rating: 4.13)
- Taman Spathodea (Predicted Rating: 4.12)

**Top 5 Recommendations for User:** 190
- Pantai Congot (Predicted Rating: 3.90)
- Taman Hiburan Rakyat (Predicted Rating: 3.82)
- Keraton Surabaya (Predicted Rating: 3.82)
- Bukit Jamur (Predicted Rating: 3.81)
- Watu Goyang (Predicted Rating: 3.81)

**Top 5 Recommendations for User:** 191
- Keraton Surabaya (Predicted Rating: 3.88)
- Glamping Lakeside Rancabali (Predicted Rating: 3.84)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.80)
- Air Terjun Kedung Pedut (Predicted Rating: 3.78)
- Bukit Wisata Pulepayung (Predicted Rating: 3.69)

**Top 5 Recommendations for User:** 192
- Curug Batu Templek (Predicted Rating: 3.97)
- Keraton Surabaya (Predicted Rating: 3.94)
- Bukit Jamur (Predicted Rating: 3.77)
- Kota Mini (Predicted Rating: 3.77)
- Water Blaster Bukit Candi Golf (Predicted Rating: 3.75)

**Top 5 Recommendations for User:** 193
- Ciwangun Indah Camp Official (Predicted Rating: 3.68)
- Teras Cikapundung BBWS (Predicted Rating: 3.67)
- Keraton Surabaya (Predicted Rating: 3.64)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.64)
- Ledok Sambi (Predicted Rating: 3.64)

**Top 5 Recommendations for User:** 194
- Monumen Selamat Datang (Predicted Rating: 3.87)
- Obyek Wisata Goa Kreo (Predicted Rating: 3.81)
- Wisata Lereng Kelir (Predicted Rating: 3.75)
- Taman Keputran (Predicted Rating: 3.73)
- Curug Batu Templek (Predicted Rating: 3.70)

**Top 5 Recommendations for User:** 195
- Taman Pelangi Yogyakarta (Predicted Rating: 3.91)
- Ocean Ecopark (Predicted Rating: 3.73)
- Upside Down World Bandung (Predicted Rating: 3.64)
- Keraton Surabaya (Predicted Rating: 3.62)
- Tafso Barn (Predicted Rating: 3.60)

**Top 5 Recommendations for User:** 196
- Obyek Wisata Goa Kreo (Predicted Rating: 4.11)
- Keraton Surabaya (Predicted Rating: 3.98)
- Taman Lansia (Predicted Rating: 3.97)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.77)
- Jogja Exotarium (Predicted Rating: 3.67)

**Top 5 Recommendations for User:** 197
- Keraton Surabaya (Predicted Rating: 4.11)
- Glamping Lakeside Rancabali (Predicted Rating: 4.11)
- Air Terjun Kedung Pedut (Predicted Rating: 4.04)
- Ledok Sambi (Predicted Rating: 4.03)
- Kampung Cina (Predicted Rating: 3.98)

**Top 5 Recommendations for User:** 198
- Keraton Surabaya (Predicted Rating: 3.97)
- Taman Sungai Mudal (Predicted Rating: 3.88)
- Monumen Yogya Kembali (Predicted Rating: 3.88)
- Desa Wisata Gamplong (Predicted Rating: 3.68)
- Tafso Barn (Predicted Rating: 3.67)

**Top 5 Recommendations for User:** 199
- Keraton Surabaya (Predicted Rating: 3.87)
- Pantai Baron (Predicted Rating: 3.75)
- Kampung Cina (Predicted Rating: 3.73)
- Sumur Gumuling (Predicted Rating: 3.72)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.71)

**Top 5 Recommendations for User:** 200
- Pantai Baron (Predicted Rating: 3.70)
- Jogja Exotarium (Predicted Rating: 3.62)
- Surabaya North Quay (Predicted Rating: 3.59)
- Taman Keputran (Predicted Rating: 3.58)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.53)

**Top 5 Recommendations for User:** 201
- Keraton Surabaya (Predicted Rating: 4.19)
- Glamping Lakeside Rancabali (Predicted Rating: 4.10)
- Air Terjun Kedung Pedut (Predicted Rating: 3.97)
- Taman Srigunting (Predicted Rating: 3.89)
- Alive Museum Ancol (Predicted Rating: 3.82)

**Top 5 Recommendations for User:** 202
- Pintoe Langit Dahromo (Predicted Rating: 4.21)
- Bukit Bintang Yogyakarta (Predicted Rating: 4.01)
- Glamping Lakeside Rancabali (Predicted Rating: 4.01)
- Kampung Wisata Taman Sari (Predicted Rating: 3.99)
- Rumah Sipitung (Predicted Rating: 3.97)

**Top 5 Recommendations for User:** 203
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.88)
- Bukit Jamur (Predicted Rating: 3.78)
- Pantai Baron (Predicted Rating: 3.72)
- Teras Cikapundung BBWS (Predicted Rating: 3.69)
- Goa Pindul (Predicted Rating: 3.68)

**Top 5 Recommendations for User:** 204
- Bukit Jamur (Predicted Rating: 4.02)
- Taman Keputran (Predicted Rating: 3.80)
- Air Terjun Kedung Pedut (Predicted Rating: 3.75)
- Wisata Lereng Kelir (Predicted Rating: 3.74)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.74)

**Top 5 Recommendations for User:** 205
- Keraton Surabaya (Predicted Rating: 4.02)
- Bukit Jamur (Predicted Rating: 3.95)
- Desa Wisata Gamplong (Predicted Rating: 3.93)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.90)
- Wisata Lereng Kelir (Predicted Rating: 3.88)

**Top 5 Recommendations for User:** 206
- Bukit Jamur (Predicted Rating: 4.13)
- Dago Dreampark (Predicted Rating: 3.92)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.89)
- Kampung Wisata Rejowinangun (Predicted Rating: 3.83)
- Pantai Watu Kodok (Predicted Rating: 3.81)

**Top 5 Recommendations for User:** 207
- Obyek Wisata Goa Kreo (Predicted Rating: 3.89)
- Kampung Cina (Predicted Rating: 3.76)
- Curug Cilengkrang (Predicted Rating: 3.74)
- Keraton Surabaya (Predicted Rating: 3.71)
- Teras Cikapundung BBWS (Predicted Rating: 3.70)

**Top 5 Recommendations for User:** 208
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.31)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.24)
- Teras Cikapundung BBWS (Predicted Rating: 4.20)
- La Kana Chapel (Predicted Rating: 4.08)
- Kampung Wisata Taman Sari (Predicted Rating: 4.07)

**Top 5 Recommendations for User:** 209
- Keraton Surabaya (Predicted Rating: 4.05)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.77)
- Observatorium Bosscha (Predicted Rating: 3.77)
- Monumen Selamat Datang (Predicted Rating: 3.74)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.69)

**Top 5 Recommendations for User:** 210
- Kampung Cina (Predicted Rating: 3.65)
- Air Terjun Kedung Pedut (Predicted Rating: 3.64)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.63)
- Monumen Yogya Kembali (Predicted Rating: 3.58)
- Sumur Gumuling (Predicted Rating: 3.57)

**Top 5 Recommendations for User:** 211
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.94)
- Kampung Cina (Predicted Rating: 3.89)
- Wisata Alam Mangrove Angke (Predicted Rating: 3.80)
- Bumi Perkemahan Cibubur (Predicted Rating: 3.79)
- Keraton Surabaya (Predicted Rating: 3.76)

**Top 5 Recommendations for User:** 212
- Glamping Lakeside Rancabali (Predicted Rating: 4.07)
- Keraton Surabaya (Predicted Rating: 4.05)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.92)
- Goa Pindul (Predicted Rating: 3.89)
- Situs Warungboto (Predicted Rating: 3.86)

**Top 5 Recommendations for User:** 213
- Taman Keputran (Predicted Rating: 4.04)
- Monumen Sanapati (Predicted Rating: 3.98)
- Rumah Sipitung (Predicted Rating: 3.88)
- Tafso Barn (Predicted Rating: 3.86)
- Gereja Perawan Maria Tak Berdosa Surabaya (Predicted Rating: 3.81)

**Top 5 Recommendations for User:** 214
- Desa Wisata Gamplong (Predicted Rating: 4.09)
- Alive Museum Ancol (Predicted Rating: 4.07)
- Grojogan Watu Purbo Bangunrejo (Predicted Rating: 3.90)
- Bukit Jamur (Predicted Rating: 3.89)
- Keraton Surabaya (Predicted Rating: 3.84)

**Top 5 Recommendations for User:** 215
- Keraton Surabaya (Predicted Rating: 3.88)
- Kampung Cina (Predicted Rating: 3.82)
- Rainbow Garden (Predicted Rating: 3.81)
- Monumen Nasional (Predicted Rating: 3.81)
- Bukit Wisata Pulepayung (Predicted Rating: 3.76)

**Top 5 Recommendations for User:** 216
- Keraton Surabaya (Predicted Rating: 4.04)
- Kampung Wisata Taman Sari (Predicted Rating: 4.02)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.97)
- Jogja Exotarium (Predicted Rating: 3.89)
- Museum Wayang (Predicted Rating: 3.86)

**Top 5 Recommendations for User:** 217
- Curug Tilu Leuwi Opat (Predicted Rating: 3.77)
- Kampung Cina (Predicted Rating: 3.64)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.61)
- Jogja Exotarium (Predicted Rating: 3.54)
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 3.47)

**Top 5 Recommendations for User:** 218
- Keraton Surabaya (Predicted Rating: 3.62)
- Jogja Exotarium (Predicted Rating: 3.61)
- Pantai Watu Kodok (Predicted Rating: 3.56)
- Kampung Cina (Predicted Rating: 3.50)
- Teras Cikapundung BBWS (Predicted Rating: 3.49)

**Top 5 Recommendations for User:** 219
- Keraton Surabaya (Predicted Rating: 3.83)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.73)
- Desa Wisata Gamplong (Predicted Rating: 3.71)
- Monumen Jalesveva Jayamahe (Predicted Rating: 3.67)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.67)

**Top 5 Recommendations for User:** 220
- Grojogan Watu Purbo Bangunrejo (Predicted Rating: 3.52)
- Candi Ijo (Predicted Rating: 3.45)
- Bukit Jamur (Predicted Rating: 3.36)
- Pantai Ngrawe (Mesra) (Predicted Rating: 3.33)
- Atlantis Water Adventure (Predicted Rating: 3.29)

**Top 5 Recommendations for User:** 221
- Jogja Exotarium (Predicted Rating: 3.79)
- Bumi Perkemahan Cibubur (Predicted Rating: 3.54)
- Taman Pelangi Yogyakarta (Predicted Rating: 3.52)
- Bukit Jamur (Predicted Rating: 3.52)
- Monumen Yogya Kembali (Predicted Rating: 3.51)

**Top 5 Recommendations for User:** 222
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.88)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.85)
- Pantai Baron (Predicted Rating: 3.83)
- Jogja Exotarium (Predicted Rating: 3.81)
- Keraton Surabaya (Predicted Rating: 3.79)

**Top 5 Recommendations for User:** 223
- Bukit Jamur (Predicted Rating: 4.20)
- Pasar Kebon Empring Bintaran (Predicted Rating: 3.92)
- Monumen Yogya Kembali (Predicted Rating: 3.87)
- Keraton Surabaya (Predicted Rating: 3.79)
- Taman Hiburan Rakyat (Predicted Rating: 3.79)

**Top 5 Recommendations for User:** 224
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.07)
- Ciwangun Indah Camp Official (Predicted Rating: 3.99)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.90)
- GunungTangkuban perahu (Predicted Rating: 3.87)
- Monumen Selamat Datang (Predicted Rating: 3.86)

**Top 5 Recommendations for User:** 225
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.04)
- Air Terjun Kedung Pedut (Predicted Rating: 4.03)
- Monumen Selamat Datang (Predicted Rating: 4.03)
- Kampung Wisata Taman Sari (Predicted Rating: 3.99)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.96)

**Top 5 Recommendations for User:** 226
- Taman Keputran (Predicted Rating: 4.17)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 4.10)
- Monumen Selamat Datang (Predicted Rating: 4.07)
- Bukit Jamur (Predicted Rating: 3.99)
- Teras Cikapundung BBWS (Predicted Rating: 3.96)

**Top 5 Recommendations for User:** 227
- Monumen Nasional (Predicted Rating: 3.97)
- Museum Nike Ardilla (Predicted Rating: 3.93)
- Keraton Surabaya (Predicted Rating: 3.92)
- Monumen Yogya Kembali (Predicted Rating: 3.91)
- Rainbow Garden (Predicted Rating: 3.90)

**Top 5 Recommendations for User:** 228
- Keraton Surabaya (Predicted Rating: 4.15)
- Bumi Perkemahan Cibubur (Predicted Rating: 4.11)
- Taman Keputran (Predicted Rating: 4.07)
- Curug Batu Templek (Predicted Rating: 4.04)
- Monumen Kapal Selam (Predicted Rating: 4.01)

**Top 5 Recommendations for User:** 229
- Pasar Kebon Empring Bintaran (Predicted Rating: 3.87)
- Bukit Jamur (Predicted Rating: 3.63)
- Taman Film (Predicted Rating: 3.63)
- Taman Pelangi (Predicted Rating: 3.54)
- Curug Batu Templek (Predicted Rating: 3.53)

**Top 5 Recommendations for User:** 230
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.06)
- Trans Studio Bandung (Predicted Rating: 3.96)
- Bukit Jamur (Predicted Rating: 3.82)
- Jogja Exotarium (Predicted Rating: 3.82)
- Monumen Yogya Kembali (Predicted Rating: 3.81)

**Top 5 Recommendations for User:** 231
- Pantai Baron (Predicted Rating: 3.82)
- Situs Warungboto (Predicted Rating: 3.75)
- Candi Gedong Songo (Predicted Rating: 3.71)
- Puspa Iptek Sundial (Predicted Rating: 3.71)
- Keraton Yogyakarta (Predicted Rating: 3.68)

**Top 5 Recommendations for User:** 232
- Pantai Baron (Predicted Rating: 3.94)
- Bukit Jamur (Predicted Rating: 3.86)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 3.86)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.76)
- Wisata Lereng Kelir (Predicted Rating: 3.76)

**Top 5 Recommendations for User:** 233
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.08)
- Taman Keputran (Predicted Rating: 3.91)
- Teras Cikapundung BBWS (Predicted Rating: 3.90)
- Gereja Tiberias Indonesia Bandung (Predicted Rating: 3.87)
- Pantai Baron (Predicted Rating: 3.85)

**Top 5 Recommendations for User:** 234
- Tafso Barn (Predicted Rating: 3.75)
- Nol Kilometer Jl.Malioboro (Predicted Rating: 3.65)
- Jogja Exotarium (Predicted Rating: 3.61)
- Pantai Baron (Predicted Rating: 3.60)
- Taman Pelangi Yogyakarta (Predicted Rating: 3.58)

**Top 5 Recommendations for User:** 235
- Tafso Barn (Predicted Rating: 4.09)
- Keraton Surabaya (Predicted Rating: 3.97)
- Margasatwa Muara Angke (Predicted Rating: 3.83)
- Sanghyang Heuleut (Predicted Rating: 3.78)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.75)

**Top 5 Recommendations for User:** 236
- Glamping Lakeside Rancabali (Predicted Rating: 3.76)
- Sanghyang Heuleut (Predicted Rating: 3.71)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.63)
- Kampung Cina (Predicted Rating: 3.63)
- Rainbow Garden (Predicted Rating: 3.62)

**Top 5 Recommendations for User:** 237
- Curug Cilengkrang (Predicted Rating: 3.85)
- Bukit Jamur (Predicted Rating: 3.65)
- Pantai Kesirat (Predicted Rating: 3.60)
- Desa Wisata Gamplong (Predicted Rating: 3.60)
- Museum Wayang (Predicted Rating: 3.57)

**Top 5 Recommendations for User:** 238
- Keraton Surabaya (Predicted Rating: 4.12)
- Jogja Exotarium (Predicted Rating: 4.05)
- Taman Spathodea (Predicted Rating: 4.00)
- Taman Hiburan Rakyat (Predicted Rating: 3.99)
- Pantai Kesirat (Predicted Rating: 3.98)

**Top 5 Recommendations for User:** 239
- Taman Lansia (Predicted Rating: 4.01)
- Gereja Perawan Maria Tak Berdosa Surabaya (Predicted Rating: 4.00)
- Desa Wisata Sungai Code Jogja Kota (Predicted Rating: 3.96)
- Kampung Wisata Taman Sari (Predicted Rating: 3.93)
- Air Terjun Kedung Pedut (Predicted Rating: 3.88)

**Top 5 Recommendations for User:** 240
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.79)
- Ledok Sambi (Predicted Rating: 3.77)
- Desa Wisata Gamplong (Predicted Rating: 3.75)
- Keraton Surabaya (Predicted Rating: 3.74)
- Bukit Jamur (Predicted Rating: 3.73)

**Top 5 Recommendations for User:** 241
- Monumen Selamat Datang (Predicted Rating: 3.89)
- Keraton Surabaya (Predicted Rating: 3.82)
- Pantai Kesirat (Predicted Rating: 3.81)
- Taman Sungai Mudal (Predicted Rating: 3.73)
- Bukit Wisata Pulepayung (Predicted Rating: 3.70)

**Top 5 Recommendations for User:** 242
- Keraton Surabaya (Predicted Rating: 4.45)
- Bukit Jamur (Predicted Rating: 4.38)
- Taman Srigunting (Predicted Rating: 4.29)
- Kampung Cina (Predicted Rating: 4.23)
- Monumen Selamat Datang (Predicted Rating: 4.21)

**Top 5 Recommendations for User:** 243
- Air Terjun Kedung Pedut (Predicted Rating: 4.22)
- Keraton Surabaya (Predicted Rating: 4.09)
- Situs Warungboto (Predicted Rating: 3.94)
- Desa Wisata Gamplong (Predicted Rating: 3.79)
- Glamping Lakeside Rancabali (Predicted Rating: 3.74)

**Top 5 Recommendations for User:** 244
- Pantai Baron (Predicted Rating: 3.95)
- Kampung Batu Malakasari (Predicted Rating: 3.86)
- Sanghyang Heuleut (Predicted Rating: 3.75)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.70)
- Keraton Surabaya (Predicted Rating: 3.68)

**Top 5 Recommendations for User:** 245
- Keraton Surabaya (Predicted Rating: 3.98)
- Ledok Sambi (Predicted Rating: 3.70)
- Puspa Iptek Sundial (Predicted Rating: 3.58)
- Sanghyang Heuleut (Predicted Rating: 3.55)
- Museum Mpu Tantular (Predicted Rating: 3.54)

**Top 5 Recommendations for User:** 246
- Curug Tilu Leuwi Opat (Predicted Rating: 3.77)
- Candi Ijo (Predicted Rating: 3.73)
- Monumen Yogya Kembali (Predicted Rating: 3.72)
- Bukit Jamur (Predicted Rating: 3.72)
- Sanghyang Heuleut (Predicted Rating: 3.65)

**Top 5 Recommendations for User:** 247
- GunungTangkuban perahu (Predicted Rating: 4.15)
- Nol Kilometer Jl.Malioboro (Predicted Rating: 4.00)
- Bukit Jamur (Predicted Rating: 3.91)
- Keraton Surabaya (Predicted Rating: 3.90)
- Kampung Cina (Predicted Rating: 3.88)

**Top 5 Recommendations for User:** 248
- Alive Museum Ancol (Predicted Rating: 4.48)
- Taman Ekspresi Dan Perpustakaan (Predicted Rating: 4.39)
- Desa Wisata Gamplong (Predicted Rating: 4.30)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.27)
- Bukit Jamur (Predicted Rating: 4.25)

**Top 5 Recommendations for User:** 249
- Museum Mpu Tantular (Predicted Rating: 3.40)
- Taman Srigunting (Predicted Rating: 3.15)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.09)
- Monumen Yogya Kembali (Predicted Rating: 3.08)
- Wisata Alam Mangrove Angke (Predicted Rating: 3.08)

**Top 5 Recommendations for User:** 250
- Keraton Surabaya (Predicted Rating: 4.20)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 4.14)
- Situs Warungboto (Predicted Rating: 4.11)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.08)
- Bukit Jamur (Predicted Rating: 4.07)

**Top 5 Recommendations for User:** 251
- Keraton Surabaya (Predicted Rating: 3.86)
- Bukit Jamur (Predicted Rating: 3.73)
- Rainbow Garden (Predicted Rating: 3.70)
- Ekowisata Mangrove Wonorejo (Predicted Rating: 3.65)
- Goa Pindul (Predicted Rating: 3.59)

**Top 5 Recommendations for User:** 252
- Grojogan Watu Purbo Bangunrejo (Predicted Rating: 3.98)
- Kampung Wisata Taman Sari (Predicted Rating: 3.94)
- Ciwangun Indah Camp Official (Predicted Rating: 3.91)
- Dago Dreampark (Predicted Rating: 3.88)
- Glamping Lakeside Rancabali (Predicted Rating: 3.83)

**Top 5 Recommendations for User:** 253
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.68)
- Teras Cikapundung BBWS (Predicted Rating: 3.60)
- Pantai Baron (Predicted Rating: 3.56)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.56)
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 3.55)

**Top 5 Recommendations for User:** 254
- Keraton Surabaya (Predicted Rating: 3.94)
- Monumen Nasional (Predicted Rating: 3.86)
- Ciwangun Indah Camp Official (Predicted Rating: 3.79)
- Monumen Yogya Kembali (Predicted Rating: 3.71)
- Monumen Selamat Datang (Predicted Rating: 3.69)

**Top 5 Recommendations for User:** 255
- Rainbow Garden (Predicted Rating: 4.21)
- Teras Cikapundung BBWS (Predicted Rating: 3.89)
- Grand Maerakaca (Predicted Rating: 3.88)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.86)
- Taman Spathodea (Predicted Rating: 3.84)

**Top 5 Recommendations for User:** 256
- Keraton Surabaya (Predicted Rating: 4.13)
- Selasar Sunaryo Art Space (Predicted Rating: 4.02)
- Taman Hutan Raya Ir. H. Juanda (Predicted Rating: 4.01)
- Monumen Nasional (Predicted Rating: 3.81)
- Monumen Yogya Kembali (Predicted Rating: 3.78)

**Top 5 Recommendations for User:** 257
- Monumen Sanapati (Predicted Rating: 3.77)
- Keraton Surabaya (Predicted Rating: 3.66)
- Kampung Cina (Predicted Rating: 3.65)
- Bukit Wisata Pulepayung (Predicted Rating: 3.64)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.64)

**Top 5 Recommendations for User:** 258
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.28)
- Bukit Wisata Pulepayung (Predicted Rating: 4.14)
- Rainbow Garden (Predicted Rating: 4.00)
- Situs Warungboto (Predicted Rating: 3.99)
- Watu Goyang (Predicted Rating: 3.98)

**Top 5 Recommendations for User:** 259
- Keraton Surabaya (Predicted Rating: 3.86)
- Bukit Jamur (Predicted Rating: 3.83)
- Desa Wisata Gamplong (Predicted Rating: 3.71)
- Taman Hiburan Rakyat (Predicted Rating: 3.66)
- Pantai Baron (Predicted Rating: 3.66)

**Top 5 Recommendations for User:** 260
- Dago Dreampark (Predicted Rating: 4.11)
- Keraton Surabaya (Predicted Rating: 4.02)
- Glamping Lakeside Rancabali (Predicted Rating: 4.01)
- Air Terjun Kedung Pedut (Predicted Rating: 3.99)
- Pantai Baron (Predicted Rating: 3.94)

**Top 5 Recommendations for User:** 261
- Keraton Surabaya (Predicted Rating: 3.68)
- Curug Tilu Leuwi Opat (Predicted Rating: 3.61)
- Teras Cikapundung BBWS (Predicted Rating: 3.54)
- Museum Tekstil (Predicted Rating: 3.51)
- Observatorium Bosscha (Predicted Rating: 3.47)

**Top 5 Recommendations for User:** 262
- Pantai Baron (Predicted Rating: 3.98)
- Glamping Lakeside Rancabali (Predicted Rating: 3.97)
- Tafso Barn (Predicted Rating: 3.93)
- Curug Tilu Leuwi Opat (Predicted Rating: 3.87)
- Monumen Yogya Kembali (Predicted Rating: 3.86)

**Top 5 Recommendations for User:** 263
- Teras Cikapundung BBWS (Predicted Rating: 4.09)
- Bukit Jamur (Predicted Rating: 3.94)
- Rumah Batik (Predicted Rating: 3.91)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.90)
- Bukit Moko (Predicted Rating: 3.78)

**Top 5 Recommendations for User:** 264
- Keraton Surabaya (Predicted Rating: 3.89)
- Curug Tilu Leuwi Opat (Predicted Rating: 3.81)
- Kampung Wisata Kadipaten (Predicted Rating: 3.72)
- Observatorium Bosscha (Predicted Rating: 3.69)
- Taman Badak (Predicted Rating: 3.66)

**Top 5 Recommendations for User:** 265
- Keraton Surabaya (Predicted Rating: 3.72)
- Kampung Cina (Predicted Rating: 3.62)
- Bukit Jamur (Predicted Rating: 3.58)
- Monumen Nasional (Predicted Rating: 3.56)
- Air Terjun Kedung Pedut (Predicted Rating: 3.48)

**Top 5 Recommendations for User:** 266
- Keraton Surabaya (Predicted Rating: 4.17)
- Taman Lansia (Predicted Rating: 4.07)
- Pantai Congot (Predicted Rating: 4.06)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.99)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.85)

**Top 5 Recommendations for User:** 267
- Keraton Surabaya (Predicted Rating: 3.94)
- Air Terjun Kedung Pedut (Predicted Rating: 3.85)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.80)
- Taman Keputran (Predicted Rating: 3.73)
- Kampung Wisata Taman Sari (Predicted Rating: 3.68)

**Top 5 Recommendations for User:** 268
- Desa Wisata Gamplong (Predicted Rating: 3.88)
- Sanghyang Heuleut (Predicted Rating: 3.88)
- Bukit Wisata Pulepayung (Predicted Rating: 3.87)
- Kampung Cina (Predicted Rating: 3.87)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.84)

**Top 5 Recommendations for User:** 269
- Kampung Wisata Taman Sari (Predicted Rating: 3.60)
- Curug Batu Templek (Predicted Rating: 3.52)
- Watu Lumbung (Predicted Rating: 3.48)
- Sanghyang Heuleut (Predicted Rating: 3.41)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.39)

**Top 5 Recommendations for User:** 270
- Teras Cikapundung BBWS (Predicted Rating: 3.73)
- Kampung Cina (Predicted Rating: 3.66)
- Pantai Baron (Predicted Rating: 3.58)
- Monumen Jalesveva Jayamahe (Predicted Rating: 3.57)
- Bukit Jamur (Predicted Rating: 3.50)

**Top 5 Recommendations for User:** 271
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.95)
- Taman Srigunting (Predicted Rating: 3.92)
- Keraton Surabaya (Predicted Rating: 3.87)
- Desa Wisata Pulesari (Predicted Rating: 3.86)
- Monumen Yogya Kembali (Predicted Rating: 3.84)

**Top 5 Recommendations for User:** 272
- Monumen Yogya Kembali (Predicted Rating: 3.91)
- Taman Pelangi Yogyakarta (Predicted Rating: 3.81)
- Keraton Surabaya (Predicted Rating: 3.81)
- Monumen Nasional (Predicted Rating: 3.79)
- Air Terjun Kedung Pedut (Predicted Rating: 3.72)

**Top 5 Recommendations for User:** 273
- Keraton Surabaya (Predicted Rating: 3.89)
- Pantai Baron (Predicted Rating: 3.88)
- Tafso Barn (Predicted Rating: 3.80)
- Teras Cikapundung BBWS (Predicted Rating: 3.79)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.79)

**Top 5 Recommendations for User:** 274
- Bukit Bintang Yogyakarta (Predicted Rating: 4.18)
- Taman Hiburan Rakyat (Predicted Rating: 4.16)
- Trans Studio Bandung (Predicted Rating: 4.15)
- Keraton Surabaya (Predicted Rating: 4.14)
- Pantai Baron (Predicted Rating: 4.05)

**Top 5 Recommendations for User:** 275
- Puspa Iptek Sundial (Predicted Rating: 3.16)
- Keraton Surabaya (Predicted Rating: 3.15)
- Desa Wisata Gamplong (Predicted Rating: 3.05)
- Bukit Jamur (Predicted Rating: 3.04)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.04)

**Top 5 Recommendations for User:** 276
- Sumur Gumuling (Predicted Rating: 3.65)
- Tafso Barn (Predicted Rating: 3.60)
- Taman Budaya Yogyakarta (Predicted Rating: 3.51)
- Monumen Yogya Kembali (Predicted Rating: 3.51)
- Monumen Selamat Datang (Predicted Rating: 3.46)

**Top 5 Recommendations for User:** 277
- Tafso Barn (Predicted Rating: 3.86)
- Kampung Cina (Predicted Rating: 3.83)
- Grand Maerakaca (Predicted Rating: 3.83)
- Desa Wisata Gamplong (Predicted Rating: 3.77)
- Jogja Bay Pirates Adventure Waterpark (Predicted Rating: 3.73)

**Top 5 Recommendations for User:** 278
- Monumen Yogya Kembali (Predicted Rating: 3.53)
- Keraton Surabaya (Predicted Rating: 3.37)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.36)
- Jogja Exotarium (Predicted Rating: 3.33)
- Sanghyang Heuleut (Predicted Rating: 3.30)

**Top 5 Recommendations for User:** 279
- Bukit Jamur (Predicted Rating: 3.99)
- Teras Cikapundung BBWS (Predicted Rating: 3.68)
- Desa Wisata Gamplong (Predicted Rating: 3.67)
- Keraton Surabaya (Predicted Rating: 3.65)
- Museum Bank Indonesia (Predicted Rating: 3.65)

**Top 5 Recommendations for User:** 280
- Atlantis Land Surabaya (Predicted Rating: 3.76)
- Monumen Yogya Kembali (Predicted Rating: 3.74)
- Keraton Surabaya (Predicted Rating: 3.73)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.69)
- Taman Pelangi Yogyakarta (Predicted Rating: 3.68)

**Top 5 Recommendations for User:** 281
- Curug Batu Templek (Predicted Rating: 3.74)
- Selasar Sunaryo Art Space (Predicted Rating: 3.73)
- Rainbow Garden (Predicted Rating: 3.70)
- Keraton Surabaya (Predicted Rating: 3.65)
- Air Terjun Kedung Pedut (Predicted Rating: 3.64)

**Top 5 Recommendations for User:** 282
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.93)
- Keraton Surabaya (Predicted Rating: 3.80)
- Monumen Nasional (Predicted Rating: 3.72)
- Kampung Korea Bandung (Predicted Rating: 3.70)
- Puspa Iptek Sundial (Predicted Rating: 3.69)

**Top 5 Recommendations for User:** 283
- Air Terjun Kedung Pedut (Predicted Rating: 4.11)
- Blue Lagoon Jogja (Predicted Rating: 3.82)
- Pintoe Langit Dahromo (Predicted Rating: 3.79)
- Grand Maerakaca (Predicted Rating: 3.79)
- Pantai Depok Jogja (Predicted Rating: 3.78)

**Top 5 Recommendations for User:** 284
- Keraton Surabaya (Predicted Rating: 3.81)
- Jogja Exotarium (Predicted Rating: 3.62)
- Masjid Agung Trans Studio Bandung (Predicted Rating: 3.61)
- Pantai Baron (Predicted Rating: 3.59)
- Bukit Jamur (Predicted Rating: 3.58)

**Top 5 Recommendations for User:** 285
- Masjid Agung Trans Studio Bandung (Predicted Rating: 4.19)
- Keraton Surabaya (Predicted Rating: 3.94)
- Teras Cikapundung BBWS (Predicted Rating: 3.90)
- Monumen Nasional (Predicted Rating: 3.88)
- Glamping Lakeside Rancabali (Predicted Rating: 3.86)

**Top 5 Recommendations for User:** 286
- Keraton Surabaya (Predicted Rating: 3.96)
- Sumur Gumuling (Predicted Rating: 3.88)
- Rumah Sipitung (Predicted Rating: 3.81)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.76)
- Alive Museum Ancol (Predicted Rating: 3.66)

**Top 5 Recommendations for User:** 287
- Keraton Surabaya (Predicted Rating: 3.82)
- Trans Studio Bandung (Predicted Rating: 3.80)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.77)
- Kampung Cina (Predicted Rating: 3.73)
- Monumen Jalesveva Jayamahe (Predicted Rating: 3.69)

**Top 5 Recommendations for User:** 288
- Kota Mini (Predicted Rating: 3.84)
- Pantai Drini (Predicted Rating: 3.82)
- Kampung Wisata Taman Sari (Predicted Rating: 3.82)
- Candi Ijo (Predicted Rating: 3.81)
- Pulau Pelangi (Predicted Rating: 3.80)

**Top 5 Recommendations for User:** 289
- Kampung Cina (Predicted Rating: 3.88)
- Selasar Sunaryo Art Space (Predicted Rating: 3.87)
- Teras Cikapundung BBWS (Predicted Rating: 3.79)
- Kota Mini (Predicted Rating: 3.67)
- Sanghyang Heuleut (Predicted Rating: 3.66)

**Top 5 Recommendations for User:** 290
- Keraton Surabaya (Predicted Rating: 3.93)
- Pantai Congot (Predicted Rating: 3.83)
- Sindu Kusuma Edupark (SKE) (Predicted Rating: 3.78)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.77)
- Kampung Wisata Taman Sari (Predicted Rating: 3.76)

**Top 5 Recommendations for User:** 291
- Bukit Jamur (Predicted Rating: 3.79)
- Keraton Surabaya (Predicted Rating: 3.67)
- Sumur Gumuling (Predicted Rating: 3.53)
- Gua Belanda (Predicted Rating: 3.50)
- Taman Hiburan Rakyat (Predicted Rating: 3.50)

**Top 5 Recommendations for User:** 292
- Wisata Alam Mangrove Angke (Predicted Rating: 4.06)
- Pantai Baron (Predicted Rating: 4.03)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.95)
- Bukit Jamur (Predicted Rating: 3.81)
- Wisata Lereng Kelir (Predicted Rating: 3.81)

**Top 5 Recommendations for User:** 293
- Taman Pelangi (Predicted Rating: 3.94)
- Observatorium Bosscha (Predicted Rating: 3.80)
- Keraton Surabaya (Predicted Rating: 3.77)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.76)
- Monumen Jalesveva Jayamahe (Predicted Rating: 3.71)

**Top 5 Recommendations for User:** 294
- Keraton Surabaya (Predicted Rating: 4.14)
- Air Terjun Kedung Pedut (Predicted Rating: 4.01)
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 3.92)
- Museum Gunung Merapi (Predicted Rating: 3.84)
- Jogja Exotarium (Predicted Rating: 3.83)

**Top 5 Recommendations for User:** 295
- Pasar Kebon Empring Bintaran (Predicted Rating: 3.81)
- Monumen Jalesveva Jayamahe (Predicted Rating: 3.73)
- Keraton Surabaya (Predicted Rating: 3.64)
- Stone Garden Citatah (Predicted Rating: 3.57)
- Bukit Bintang Yogyakarta (Predicted Rating: 3.57)

**Top 5 Recommendations for User:** 296
- Kampung Cina (Predicted Rating: 3.66)
- Bukit Jamur (Predicted Rating: 3.63)
- Old City 3D Trick Art Museum (Predicted Rating: 3.59)
- Air Terjun Kedung Pedut (Predicted Rating: 3.59)
- Keraton Surabaya (Predicted Rating: 3.58)

**Top 5 Recommendations for User:** 297
- Taman Keputran (Predicted Rating: 4.11)
- Pantai Baron (Predicted Rating: 4.05)
- Tafso Barn (Predicted Rating: 3.93)
- Atlantis Land Surabaya (Predicted Rating: 3.93)
- Puncak Gunung Api Purba - Nglanggeran (Predicted Rating: 3.91)

**Top 5 Recommendations for User:** 298
- Keraton Surabaya (Predicted Rating: 4.44)
- Teras Cikapundung BBWS (Predicted Rating: 4.27)
- Desa Wisata Gamplong (Predicted Rating: 4.27)
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 4.25)
- Alive Museum Ancol (Predicted Rating: 4.25)

**Top 5 Recommendations for User:** 299
- Dago Dreampark (Predicted Rating: 3.92)
- The World Landmarks - Merapi Park Yogyakarta (Predicted Rating: 3.72)
- Keraton Surabaya (Predicted Rating: 3.60)
- Curug Batu Templek (Predicted Rating: 3.55)
- Monumen Selamat Datang (Predicted Rating: 3.52)

**Top 5 Recommendations for User:** 300
- Desa Wisata Gamplong (Predicted Rating: 4.16)
- Taman Spathodea (Predicted Rating: 4.13)
- Rumah Sipitung (Predicted Rating: 4.01)
- Monumen Nasional (Predicted Rating: 4.00)
- Monumen Selamat Datang (Predicted Rating: 3.97)

Process finished with exit code 0
