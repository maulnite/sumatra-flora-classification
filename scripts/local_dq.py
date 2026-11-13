# Modul data + augmentasi. Ditaruh di file biar worker DataLoader (forkserver/spawn) bisa import,
# soalnya class yang didefinisiin langsung di notebook gak bisa di-pickle, dan fork + OpenCV bisa deadlock.
import cv2, numpy as np, albumentations as A, torch
from torch.utils.data import Dataset

cv2.setNumThreads(1)  # 1 thread per proses; paralelnya dari banyak worker, bukan dari OpenCV
IMG = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def read_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def squash(img, size=IMG):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def to_tensor(img):  # HWC uint8 -> CHW float ternormalisasi
    return torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1))


def poly_occluder(img, **kw):
    # poligon warna semi-transparan (kayak tempelan daun/bentuk di test, F4)
    out = img.copy(); h, w = img.shape[:2]
    for _ in range(np.random.randint(1, 4)):
        c = np.array([np.random.randint(w), np.random.randint(h)])
        pts = (c + np.random.randn(np.random.randint(3, 7), 2) * min(h, w) * 0.12).astype(np.int32)
        layer = out.copy()
        cv2.fillPoly(layer, [pts], tuple(int(x) for x in np.random.randint(0, 200, 3)))
        a = np.random.uniform(0.4, 0.9)
        out = cv2.addWeighted(layer, a, out, 1 - a, 0)
    return out


def haze(img, **kw):
    # kabut murah: blend ke abu terang. Pengganti A.RandomFog yang ~600 ms/gambar di resolusi asli
    a = np.random.uniform(0.15, 0.45)
    return cv2.addWeighted(img, 1 - a, np.full_like(img, np.random.randint(180, 240)), a, 0)


# v1: urutan awal tebakan: geometric -> warna -> cuaca/occluder -> cutout -> JPEG
corrupt = A.Compose([
    A.Affine(scale=(0.8, 1.1), translate_percent=(-0.2, 0.2), rotate=(-20, 20),
             border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=60, sat_shift_limit=60, val_shift_limit=30),
        A.RGBShift(r_shift_limit=60, g_shift_limit=60, b_shift_limit=60),
        A.ChannelShuffle(),
    ], p=0.7),
    A.ToGray(p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.6),
    A.OneOf([
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3)),
        A.Lambda(image=haze),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=200),
        A.RandomRain(drop_length=40, blur_value=3),
        A.Lambda(image=poly_occluder),
    ], p=0.6),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.05, 0.2), hole_width_range=(0.05, 0.2), fill=0, p=0.3),
    A.ImageCompression(quality_range=(10, 50), p=1.0),
])



def haze_v2(img, **kw):
    a = np.random.uniform(0.1, 0.3)
    return cv2.addWeighted(img, 1 - a, np.full_like(img, np.random.randint(180, 240)), a, 0)


# v4 (iterasi 2): dituning biar statistik warna cocok sama test. Warna dulu, baru geometric, biar border tetap hitam.
corrupt_v4 = A.Compose([
    A.OneOf([  # bobot relatif RGBShift 4 : HSV 2 : shuffle 1; shift G lebih gede -> tint diagonal ijo/magenta kayak test
        A.RGBShift(r_shift_limit=40, g_shift_limit=80, b_shift_limit=40, p=1.0),
        A.HueSaturationValue(hue_shift_limit=60, sat_shift_limit=(-20, 80), val_shift_limit=(-15, 25), p=0.5),
        A.ChannelShuffle(p=0.25),
    ], p=0.9),
    A.ColorJitter(brightness=0, contrast=0, saturation=(0.8, 1.8), hue=0, p=0.6),
    A.ToGray(p=0.1),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.35), contrast_limit=0.35, p=0.6),
    A.Affine(scale=(0.8, 1.1), translate_percent=(-0.2, 0.2), rotate=(-20, 20),
             border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.5),
    A.OneOf([
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3)),
        A.Lambda(image=haze_v2),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=200),
        A.RandomRain(drop_length=40, blur_value=3),
        A.Lambda(image=poly_occluder),
    ], p=0.4),
    A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.05, 0.2), hole_width_range=(0.05, 0.2), fill=0, p=0.3),
    A.ImageCompression(quality_range=(10, 50), p=1.0),
])
CORRUPTERS = {"corrupt": corrupt, "corrupt_v4": corrupt_v4}

# hard: v4 + korupsi lebih berat. Test ternyata lebih susah dari v4 (median confidence test 0.34 vs val v4 0.51, LB E 0.519 vs val 0.651),
# jadi tambah blur/noise/downscale (belum ada di v4), cuaca lebih sering, JPEG lebih jelek.
corrupt_hard = A.Compose([
    A.OneOf([
        A.RGBShift(r_shift_limit=40, g_shift_limit=80, b_shift_limit=40, p=1.0),
        A.HueSaturationValue(hue_shift_limit=60, sat_shift_limit=(-20, 80), val_shift_limit=(-15, 25), p=0.5),
        A.ChannelShuffle(p=0.25),
    ], p=0.9),
    A.ColorJitter(brightness=0, contrast=0, saturation=(0.8, 1.8), hue=0, p=0.6),
    A.ToGray(p=0.15),
    A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.4), contrast_limit=0.45, p=0.7),
    A.Affine(scale=(0.7, 1.1), translate_percent=(-0.25, 0.25), rotate=(-30, 30),
             border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.6),
    A.OneOf([
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3)),
        A.Lambda(image=haze_v2),
        A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=200),
        A.RandomRain(drop_length=40, blur_value=3),
        A.Lambda(image=poly_occluder),
    ], p=0.7),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 9)), A.MotionBlur(blur_limit=(3, 11)),
        A.GaussNoise(std_range=(0.05, 0.2)), A.Downscale(scale_range=(0.25, 0.5)),
    ], p=0.5),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.25), hole_width_range=(0.05, 0.25), fill=0, p=0.4),
    A.ImageCompression(quality_range=(5, 40), p=1.0),
])
CORRUPTERS["corrupt_hard"] = corrupt_hard

rrc = A.Compose([A.RandomResizedCrop(size=(IMG, IMG), scale=(0.25, 1.0), ratio=(0.5, 2.0)), A.HorizontalFlip()])
light_color = A.Compose([A.ColorJitter(0.2, 0.2, 0.2, 0.02, p=0.5)])


def val_pair(args):
    """(path, seed[, versi]) -> (versi bersih, versi terkorupsi) 224x224; seed tetap biar val deterministik."""
    path, seed, *ver = args
    c = CORRUPTERS[ver[0] if ver else "corrupt"]
    img = read_rgb(path)
    np.random.seed(seed); c.set_random_seed(seed)
    return squash(img), squash(c(image=img)["image"])


class TrainDS(Dataset):
    def __init__(self, paths, labels, mode):  # mode: "clean" (aug standar) / "corrupt" (v1) / "corrupt_v4"
        self.paths, self.labels, self.mode = list(paths), list(labels), mode
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = read_rgb(self.paths[i])
        img = CORRUPTERS[self.mode](image=img)["image"] if self.mode in CORRUPTERS else light_color(image=img)["image"]
        # uint8 CHW: 4x lebih kecil di shared memory, normalisasi di GPU (train_fold)
        return torch.from_numpy(np.ascontiguousarray(rrc(image=img)["image"].transpose(2, 0, 1))), self.labels[i]


class ArrDS(Dataset):
    def __init__(self, arr, labels=None): self.arr, self.labels = arr, labels
    def __len__(self): return len(self.arr)
    def __getitem__(self, i): return to_tensor(self.arr[i]), (-1 if self.labels is None else self.labels[i])


def worker_init(_):
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); corrupt.set_random_seed(seed); corrupt_v4.set_random_seed(seed); corrupt_hard.set_random_seed(seed); rrc.set_random_seed(seed); light_color.set_random_seed(seed)
