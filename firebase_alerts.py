# ============================================
# ملف نظام التنبيهات (firebase_alerts.py)
# ============================================
# هذا الملف يدير إرسال التنبيهات للمستخدمين عند اكتشاف حيوان
# يقوم بـ:
# - الاتصال بـ Firebase (Realtime Database و Firestore)
# - حساب المسافة بين موقع الكاميرا وموقع كل مستخدم
# - إرسال إشعارات (Push Notifications) للمستخدمين القريبين فقط
# - تحديث Realtime Database بالتنبيه النشط
# ============================================

import math
import time

import firebase_admin
from firebase_admin import credentials, db, firestore, messaging

# 1. إعداد الاتصال - مسار ملف serviceAccountKey.json (يحتوي على مفاتيح Firebase)
service_account_path = r"serviceAccountKey.json"

# تهيئة التطبيق - الاتصال بـ Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(service_account_path)
    firebase_admin.initialize_app(
        cred, {"databaseURL": "https://animal-70086-default-rtdb.europe-west1.firebasedatabase.app/"}
    )

fs = firestore.client()  # عميل Firestore للوصول إلى قاعدة البيانات

# 2. تثبيت الموقع يدوياً (الإحداثيات التي حددتها أنت) - موقع الكاميرا الثابت
CAM_LAT = 30.202947
CAM_LNG = 35.733174

# 3. نطاق الإرسال (بالكيلومتر) - يمكنك تعديله حسب احتياجك
# سيتم إرسال التنبيهات فقط للمستخدمين الذين يقعون ضمن هذا النطاق
ALERT_RADIUS_KM = 1  # إرسال التنبيهات للمستخدمين ضمن 1 كم من الكاميرا

# 4. إعدادات الإرسال
# عندما يتم تفعيل حفظ الموقع في React Native، غيّر هذا إلى False
SEND_TO_USERS_WITHOUT_LOCATION = True  # إذا True: يرسل للجميع حتى بدون موقع (مؤقت حتى يتم حفظ الموقع)

print(f"📍 تم تثبيت موقع الكاميرا على الإحداثيات: {CAM_LAT}, {CAM_LNG}")
print(f"📡 سيتم إرسال التنبيهات للمستخدمين ضمن {ALERT_RADIUS_KM} كم من الكاميرا")
if SEND_TO_USERS_WITHOUT_LOCATION:
    print("⚠️ وضع التجربة: سيتم الإرسال للمستخدمين بدون موقع أيضاً")


# دالة حساب المسافة بين نقطتين جغرافيتين - تستخدم معادلة Haversine
def calculate_distance(lat1, lon1, lat2, lon2):
    """حساب المسافة بين إحداثيين جغرافيين باستخدام معادلة Haversine الإرجاع: المسافة بالكيلومتر."""
    # نصف قطر الأرض بالكيلومتر
    R = 6371.0

    # تحويل الدرجات إلى الراديان
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # الفرق في الإحداثيات
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # معادلة Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


# الدالة الرئيسية - إرسال التنبيهات للمستخدمين القريبين
def broadcast_animal_alert(animal_type, distance_from_cam):
    """إرسال التنبيهات للمستخدمين القريبين فقط من موقع الكاميرا."""
    animal_formatted = animal_type.capitalize()
    lat, lng = float(CAM_LAT), float(CAM_LNG)
    dist = float(distance_from_cam)

    try:
        # أ. تحديث Realtime Database - تحديث التنبيه النشط (يقرأه التطبيق مباشرة)
        alert_ref = db.reference("test/Alarts")
        alert_ref.update(
            {
                "type": animal_formatted,
                "Distance": dist,
                "active": True,
                "latitude": lat,
                "longitude": lng,
                "timestamp": int(time.time() * 1000),
            }
        )

        # ب. إضافة سجل في Firestore للـ History (يمكن إضافتها لاحقاً)

        # ج. إرسال الإشعارات (FCM) للمستخدمين القريبين فقط - البحث عن المستخدمين ضمن النطاق
        nearby_tokens = []
        total_users = 0
        users_without_location = 0
        users_without_token = 0

        print("🔍 البحث عن المستخدمين القريبين...")

        # قراءة جميع المستخدمين وفلترة حسب المسافة - جلب جميع المستخدمين من Firestore
        for doc in fs.collection("users").stream():
            user_data = doc.to_dict()
            total_users += 1
            user_id = doc.id

            # الحصول على FCM Token
            fcm_token = user_data.get("fcmToken")
            if not fcm_token:
                users_without_token += 1
                print(f"  ⚠️ المستخدم {user_id}: لا يوجد FCM Token")
                continue

            # الحصول على موقع المستخدم
            user_lat = user_data.get("latitude")
            user_lng = user_data.get("longitude")

            # إذا لم يكن هناك موقع مسجل
            if user_lat is None or user_lng is None:
                users_without_location += 1
                print(f"  ⚠️ المستخدم {user_id}: لا يوجد موقع مسجل (lat: {user_lat}, lng: {user_lng})")
                # خيار: إرسال للجميع إذا لم يكن هناك موقع (للتجربة فقط)
                if SEND_TO_USERS_WITHOUT_LOCATION:
                    nearby_tokens.append(fcm_token)
                    print("    → تم إضافة المستخدم (وضع التجربة)")
                continue

            try:
                # حساب المسافة بين الكاميرا والمستخدم
                distance_km = calculate_distance(lat, lng, float(user_lat), float(user_lng))

                # إذا كان المستخدم ضمن النطاق المحدد، أضفه للقائمة
                if distance_km <= ALERT_RADIUS_KM:
                    nearby_tokens.append(fcm_token)
                    print(f"  ✅ مستخدم {user_id} ضمن النطاق: {distance_km:.2f} كم من الكاميرا")
                else:
                    print(f"  ❌ مستخدم {user_id} خارج النطاق: {distance_km:.2f} كم (أبعد من {ALERT_RADIUS_KM} كم)")
            except Exception as loc_error:
                print(f"  ❌ خطأ في حساب المسافة للمستخدم {user_id}: {loc_error}")

        print("\n📊 ملخص البحث:")
        print(f"  - إجمالي المستخدمين: {total_users}")
        print(f"  - مستخدمين بدون موقع: {users_without_location}")
        print(f"  - مستخدمين بدون Token: {users_without_token}")
        print(f"  - مستخدمين ضمن النطاق: {len(nearby_tokens)}")

        # إرسال الإشعارات للمستخدمين القريبين
        if nearby_tokens:
            try:
                messages = [
                    messaging.Message(
                        notification=messaging.Notification(
                            title="⚠️ تحذير من الطريق",
                            body=f"تم رصد {animal_formatted} على بعد {dist:.0f} متر.",
                        ),
                        token=token,
                    )
                    for token in nearby_tokens
                ]

                # إرسال الرسائل مع معالجة الأخطاء - إرسال الإشعارات لجميع المستخدمين القريبين
                batch_response = messaging.send_each(messages)

                # التحقق من النتائج - عد الإشعارات المرسلة بنجاح والفاشلة
                success_count = 0
                failure_count = 0
                for i, response in enumerate(batch_response.responses):
                    if response.success:
                        success_count += 1
                    else:
                        failure_count += 1
                        print(f"  ❌ فشل إرسال للمستخدم {i + 1}: {response.exception}")

                print(f"✅ تم إرسال التنبيه بنجاح: {success_count} نجح, {failure_count} فشل")
                print(f"✅ إجمالي الرسائل المرسلة: {len(nearby_tokens)} من أصل {total_users} مستخدم.")
            except Exception as send_error:
                print(f"❌ خطأ في إرسال الرسائل: {send_error}")
                import traceback

                traceback.print_exc()
        else:
            print(f"⚠️ لا يوجد مستخدمين قريبين ضمن {ALERT_RADIUS_KM} كم لإرسال الإشعارات.")
            print("💡 نصيحة: تأكد من أن المستخدمين لديهم 'latitude' و 'longitude' في Firestore.")

    except Exception as e:
        print(f"❌ خطأ في الإرسال: {e}")
