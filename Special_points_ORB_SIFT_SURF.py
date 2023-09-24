import time
import multiprocessing as mp
from glib import *
import copy

class spetial_points:
    def __init__(self, Q_2:mp.Queue, name_image, name_method):
        self.name_query_image = name_image
        self.Q_2, self.name_method = Q_2, name_method
        self.name_metod_start = self.choise_method_points(self.name_method.lower())
        self.process_method = mp.Process(target=self.name_metod_start, args=())
        self.process_method.start()

    def choise_method_points(self, name_method):
        choice_function = None
        if name_method == "orb":
            choice_function = self.method_ORB
        elif name_method == "sift":
            choice_function = self.method_SIFT
        elif name_method == "surf":
            choice_function = self.method_SURF

        return choice_function

    def method_SIFT(self):
        img1 = cv.imread(self.name_query_image, cv.IMREAD_GRAYSCALE)  # queryImage
        sift = cv.xfeatures2d.SIFT_create(100)
        kp1, des1 = sift.detectAndCompute(img1, None)
        bf = cv.BFMatcher()
        minRatio = 3 / 4
        while True:
            if not self.Q_2.empty():
                image, cordinat = self.Q_2.get()

                try:
                    for i, k in enumerate(cordinat):
                        img2 = image[k[1]:k[3], k[0]:k[2]]
                        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
                        kp2, des2 = sift.detectAndCompute(img2, None)
                        knnMatches = bf.knnMatch(des1, des2, k=2)
                        goodmatch = sorted([[m] for m, n in knnMatches if m.distance / n.distance < minRatio], key=lambda x:x[0].distance)
                        img3 = cv.drawMatches(img1, kp1, img2, kp2, goodmatch, None, flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
                        cv.imshow("frame", cv.resize(img3[:, :, ::-1], (500, 500)))
                        cv.waitKey(1)
                except:
                    print("None type")

    def method_SURF(self):
        img1 = cv.imread(self.name_query_image, cv.IMREAD_GRAYSCALE)  # queryImage
        surf = cv.xfeatures2d.SURF_create(25000)
        kp1, des1 = surf.detectAndCompute(img1, None)
        match = cv.FlannBasedMatcher()
        while True:
            if not self.Q_2.empty():
                image, cordinat = self.Q_2.get()

                try:
                    for i, k in enumerate(cordinat):
                        img2 = image[k[1]:k[3], k[0]:k[2]]
                        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
                        kp2, des2 = surf.detectAndCompute(img2, None)
                        matches = match.match(des1, des2)
                        lst_dist = [k.distance for i, k in enumerate(matches)]
                        minMatch, maxMatch = min(lst_dist), max(lst_dist)
                        goodmatch = [k for i, k in enumerate(matches) if k.distance < minMatch + (maxMatch - minMatch)/2]
                        img3 = cv.drawMatches(img1, kp1, img2, kp2, goodmatch, None, flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
                        cv.imshow("frame", cv.resize(img3, (500, 500)))
                        cv.waitKey(1)
                except:
                    print("None type")

    def method_ORB(self):
        img1 = cv.imread(self.name_query_image, cv.IMREAD_GRAYSCALE)  # queryImage
        h1, w1 = img1.shape
        orb = cv.ORB_create(nfeatures=500, fastThreshold=1, edgeThreshold=31, patchSize=31, firstLevel=0, nlevels=8, WTA_K=2)
        kp1, des1 = orb.detectAndCompute(img1, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        while True:
            if not self.Q_2.empty():
                image, cordinat = self.Q_2.get()

                try:
                    for i, k in enumerate(cordinat):
                        img2 = image[k[1]:k[3], k[0]:k[2]]
                        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
                        h2, w2 = img2.shape
                        if h2 <= h1 and w2 <= w1:
                            kp2, des2 = orb.detectAndCompute(img2, None)
                            matches = bf.match(des1, des2)
                            matches = sorted(matches, key=lambda x: x.distance)
                            if len(matches) >= 1:
                                img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None,
                                                      flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                                cv.imshow("frame", cv.resize(img3, (500, 500)))
                                cv.waitKey(1)
                except:
                    print("None type")


class usb_cam_initialization: # пересмотреть отдельно #######
    def __init__(self, number_cam):
        self.number_cam = number_cam

    def fuind_cam(self):
        c, count_unsuccess = 0, 0
        num_cam = None
        while True:

            try:
                if self.number_cam != None:
                    cam = cv.VideoCapture(self.number_cam)
                    rets, img = cam.read()
                    if rets:
                        num_cam = self.number_cam
                        print("Number_cam_it's done:", num_cam)
                        break
                    else:
                        self.number_cam = None
                        print("Not_found_cam_number_latter_screach:", self.number_cam)
                cam = cv.VideoCapture(c)
                rets, img = cam.read()
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                print("camera:", rets)
                num_cam = c
                break

            except:
                print("Error_number_camera")
                c += 1
                count_unsuccess += 1
                print("Нет подключения к камере, количество попыток подключения:", count_unsuccess)
                if count_unsuccess > 6:
                    print("Камера не найдена, выход из режима поиска доступной USB камеры")
                    break

        return num_cam

class CAM:
    def __init__(self, Q_1:mp.Queue, num, mode):
        self.num, self.mode = num, mode
        self.img_queue_1 = Q_1
        self.proc_cam = mp.Process(target=self.image_send, args=())
        self.proc_cam.start()

    def choice_videofile_or_camera(self):
        return cv.VideoCapture(usb_cam_initialization(self.num).fuind_cam()) if self.mode.lower() == "camera" else cv.VideoCapture("2_n.avi")

    def image_send(self):
        cap = self.choice_videofile_or_camera()
        while cap.isOpened():
            flag, image = cap.read()
            if self.img_queue_1.empty():
                self.img_queue_1.put(image)



class Moution_detect:
    def __init__(self, Q:mp.Queue, Q_2:mp.Queue):
        self.image_lst, self.Q_2 = Q, Q_2
        self.flag_show, self.triples = False, []
        self.image_preparing_process = mp.Process(target=self.imshow, args=())
        self.image_preparing_process.start()


    def draw_traectory(self, boxes_1, boxes_2, boxes_3, triples, image_2):
        lst_cord = []
        for triple in triples:
            box3 = boxes_3[triple[2]]
            box2 = boxes_2[triple[1]]
            box1 = boxes_1[triple[0]]
            b_to_draw = self.box_cvt_cent2corners(box3)  # преобразуем вид бокса на текущем кадре для удобства отрисовки
            lst_cord.append(b_to_draw)
            cv.rectangle(image_2, (b_to_draw[0], b_to_draw[1]), (b_to_draw[2], b_to_draw[3]), (0, 0, 0), 4)
            cv.line(image_2, (box3[0], box3[1]), (box2[0], box2[1]), (0, 0, 200), 4)
            cv.line(image_2, (box2[0], box2[1]), (box1[0], box1[1]), (0, 200, 0), 2)
            cv.circle(image_2, (box3[0], box3[1]), 50, (0, 0, 0), 10)
        return lst_cord

    def box_cvt_cent2corners(self, box):

        pt_lt = (int(box[0] - box[2] / 2), int(box[1] - box[3] / 2))
        pt_rb = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))

        return (pt_lt[0], pt_lt[1], pt_rb[0], pt_rb[1])

    def check_size_weight_height_bbox(self, bound_box_1, bound_box_2, percent=0.3, version="central_cord"):

        if version == "central_cord":
            w1, h1, w2, h2 = bound_box_1[2], bound_box_1[3], bound_box_2[2], bound_box_2[3]  # Получение ширины и высоты текущего объекта  и предшествующего ( при условии если в массив подаются координаты центра области)

        else:  # Если даны координаты левого верхнего и правого нижнего
            w1, h1, w2, h2 = bound_box_1[2] - bound_box_1[0], bound_box_1[3] - bound_box_1[1], bound_box_2[2] - \
                             bound_box_2[0], bound_box_2[3] - bound_box_2[
                                 1]  # Получение ширины и высоты текущего объекта  и предшествующего ( при иной подаче координат)

        flag_x = True if (w1 - w1 * percent) <= w2 and (w1 + w1 * percent) >= w2 else False  # проверка длины области объекта
        flag_y = True if (h1 - h1 * percent) <= h2 and (h1 + h1 * percent) >= h2 else False  # проверка ширины области объекта

        return all([flag_x, flag_y])  # Получаем True при условии что все флаги True

    def standart_proc(self, dif):  # Функция расчета делатации и эрозии

        kernel = np.ones((4, 4), np.uint8)  # Ядро эрозии
        kernel2 = np.ones((32, 32), np.uint8)  # Ядро дилатации
        thresh = 20  # Порог бинаризации
        cv.threshold(dif, thresh, 1, cv.THRESH_BINARY, dif)  # Бинаризация изображения
        '''Эрозия'''
        cv.erode(dif, kernel, dif, iterations=1)  # Функция эрозии
        '''Дилатация'''
        cv.dilate(dif, kernel2, dif, iterations=2)  # Функция дилатации

    def find_contours_and_circle(self, image, diff, detect_last, last_last_detect, color_B=0, color_G=0, color_R=0, size_line=1, radius=2):  # Функция расчета контуров и отрисовки областей текущего кадра
        Box3 = []
        сontours, _ = cv.findContours(diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Функция расчета контуров
        for i, contour in enumerate(сontours):  # Проход по координатам текущих контуров
            (x, y, w, h) = cv.boundingRect(contour)  # Функция нахождения координат прямоугольника около цели
            Box3.append([x + w // 2, y + h // 2, w, h])
            if self.flag_show:
                cv.rectangle(image, (x, y), (x + w, y + h), (color_B, color_G, color_R), size_line)  # отрисовка прямоугольника около текущей цели
                cv.circle(image, (x + w // 2, y + h // 2), radius, (color_B, color_G, color_R), 4, 1)  # отрисовка окружности вокруг текущей цели

            for key, cord_val in enumerate(detect_last):  # проход циклом по предшествующим областям (зеленые боксы)

                flags = self.check_size_weight_height_bbox([x + w // 2, y + h // 2, w, h], cord_val, percent=0.25)  # Проверка размерности областей текущего и прошедшего кадра

                if pt2pt_2d_range((cord_val[0], cord_val[1]), (x + w // 2, y + h // 2)) < radius and flags == True:  # условие которое проверяет длину отрезка который должен быть меньше установленного радиуса
                    if self.flag_show:
                        cv.line(image, (cord_val[0], cord_val[1]), (x + w // 2, y + h // 2), (0, 0, 0), 10, 2)  # соединяем линией две точки которые удовлетворяют условию меньше радиуса

                    # Предсказание положения объекта
                    vector_XYcent = [x + w // 2 - cord_val[0], y + h // 2 - cord_val[1]]  #
                    vector_pred = [0, 0, cord_val[0], cord_val[1]]  #
                    vector_pred[0], vector_pred[1] = cord_val[0] - vector_XYcent[0], cord_val[1] - vector_XYcent[1]  #
                    if self.flag_show:
                        cv.circle(image, (vector_pred[0], vector_pred[1]), 15, (0, 0, 0), 14, 2)  #

                    obl_id, obl_range = [], []

                    for j, k in enumerate(last_last_detect):
                        flag = self.check_size_weight_height_bbox([x + w // 2, y + h // 2, w, h], cord_val, percent=0.25)

                        if pt2pt_2d_range((k[0], k[1]), (vector_pred[0], vector_pred[1])) < radius and flag:
                            range = pt2pt_2d_range((vector_pred[0], vector_pred[1]), (k[0], k[1]))
                            obl_id.append(j)
                            obl_range.append(range)

                    if len(obl_range) > 0:
                        min_range = min(obl_range)
                        idx = obl_range.index(min_range)
                        b1_idx_ = obl_id[idx]
                        self.triples.append((b1_idx_, key, i))
                        print(f'min_range:{min_range}')

            return Box3

    def find_countors(self, image, diff, color_B=0, color_G=0, color_R=0, size_line=1):  # Функция расчета контуров и отрисовки областей предшествующих кадров

        сontours, _ = cv.findContours(diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  # Обнаружение контуров предшествующих областей
        detect = []  # Массив предшествующих детектирований

        for i, contour in enumerate(сontours):  # Проход по предшествующим контурам
            (x, y, w, h) = cv.boundingRect(contour)  # Нахождение координат бокса предшествующей картиники
            detect.append([x + w // 2, y + h // 2, w, h])  # Добавление координат предшествущих областей
            if self.flag_show:
                cv.rectangle(image, (x, y), (x + w, y + h), (color_B, color_G, color_R),
                             size_line)  # Отрисовка предшествующих областей

        return detect



    def imshow(self):
        image = self.image_lst.get()
        gray1 = copy.deepcopy(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

        image = self.image_lst.get()
        gray2 = copy.deepcopy(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

        image = self.image_lst.get()
        gray3 = copy.deepcopy(cv.cvtColor(image, cv.COLOR_BGR2GRAY))

        I2 = cv.absdiff(gray3, gray2)

        I1 = cv.absdiff(gray2, gray1)

        count = 0

        while True:
            if not self.image_lst.empty():
                t_start = time.time()
                image = self.image_lst.get()
                image_3 = copy.deepcopy(image)
                image_2 = copy.deepcopy(image)
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                gray1 = copy.deepcopy(image)

                gray1, gray2, gray3 = gray2, gray3, gray1

                I2new = cv.absdiff(gray3, gray2)
                I1new = cv.absdiff(gray2, gray1)
                I_d_f = cv.subtract(I2new, I2)
                I_d_f_last = cv.subtract(I2new, I_d_f)
                I_d_f_last_2 = cv.subtract(I1, I_d_f_last)
                I2 = I2new
                I1 = I1new
                self.triples = []
                self.standart_proc(I_d_f)
                self.standart_proc(I_d_f_last)
                last_detection = self.find_countors(image_2, I_d_f_last, color_G=255, size_line=4)
                self.standart_proc(I_d_f_last_2)
                last_last_detect = self.find_countors(image_2, I_d_f_last_2, color_G=255, size_line=2)

                boxes_3 = self.find_contours_and_circle(image_2, I_d_f, last_detection, last_last_detect, color_R=255, size_line=8, radius=120)
                t_middle = time.time()


                count += 1  # Счетчик кадров
                cords = self.draw_traectory(last_last_detect, last_detection, boxes_3, self.triples, image_2)
                self.Q_2.put([image_3, cords])
                cv.imshow("frame_moution", cv.resize(image_2, (600, 600)))
                cv.waitKey(1)


if __name__ == '__main__':
    Q_1, Q_2 = mp.Queue(maxsize=1), mp.Queue(maxsize=1)
    num = 5
    cam = CAM(Q_1, num, "videofile")
    moution = Moution_detect(Q_1, Q_2)
    secial_point = spetial_points(Q_2, "47.jpg", "ORB")

