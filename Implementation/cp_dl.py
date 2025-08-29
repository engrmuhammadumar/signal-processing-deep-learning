import os, numpy as np, pandas as pd, itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from cp_utils import ensure_dirs
from cp_config import IMG_SIZE, BATCH, SEED, TF_EPOCHS, TF_FT_EPOCHS, MAX_TSNE_POINTS

# -------- datasets ----------
def prepare_tf_datasets(df, target_size=IMG_SIZE, batch=BATCH):
    labels = sorted(df["label"].unique())
    label2id = {l:i for i,l in enumerate(labels)}
    df = df.copy(); df["y"] = df["label"].map(label2id)

    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["y"])
    tr, va = train_test_split(tr, test_size=0.2, random_state=SEED, stratify=tr["y"])

    def load_img(p):
        img = tf.io.read_file(p)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32)/255.0
        return img

    def make_ds(dd, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((dd["img"].tolist(), dd["y"].tolist()))
        ds = ds.map(lambda p,y: (load_img(p), tf.one_hot(y, depth=len(labels))),
                    num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle: ds = ds.shuffle(2048, seed=SEED)
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    return make_ds(tr,True), make_ds(va), make_ds(te), labels

# -------- models ---------
def make_model(arch, img_size=IMG_SIZE, n_classes=4):
    input_shape = (*img_size,3)
    x_in = keras.Input(shape=input_shape)
    arch = arch.lower()
    if arch == "vgg16":
        base = keras.applications.VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
        preprocess = keras.applications.vgg16.preprocess_input
        unfreeze_from, ft_lr = 10, 1e-5
    elif arch == "resnet50":
        base = keras.applications.ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
        preprocess = keras.applications.resnet50.preprocess_input
        unfreeze_from, ft_lr = 140, 1e-5
    elif arch == "efficientnetb0":
        base = keras.applications.EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
        preprocess = keras.applications.efficientnet.preprocess_input
        unfreeze_from, ft_lr = 200, 5e-5
    else:
        raise ValueError("arch must be vgg16 | resnet50 | efficientnetb0")

    x = layers.Lambda(preprocess)(x_in)
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu", name="penultimate")(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(x_in, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model, base, unfreeze_from, ft_lr

def fine_tune(model, base, unfreeze_from=100, lr=1e-5):
    base.trainable = True
    for i, layer in enumerate(base.layers):
        layer.trainable = (i >= unfreeze_from)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def plot_history(hist, title_prefix, out_dir):
    acc = hist.history.get("accuracy", [])
    val_acc = hist.history.get("val_accuracy", [])
    loss = hist.history.get("loss", [])
    val_loss = hist.history.get("val_loss", [])

    plt.figure(figsize=(7,4))
    plt.plot(acc, label="train"); plt.plot(val_acc, label="val")
    plt.title(f"{title_prefix} - Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{title_prefix}_acc.png"), dpi=150); plt.close()

    plt.figure(figsize=(7,4))
    plt.plot(loss, label="train"); plt.plot(val_loss, label="val")
    plt.title(f"{title_prefix} - Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{title_prefix}_loss.png"), dpi=150); plt.close()

def plot_confusion(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45); plt.yticks(tick, labels)
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],ha='center',va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def extract_penultimate(model):
    return keras.Model(inputs=model.input, outputs=model.get_layer("penultimate").output)

def run_tsne(model, ds, labels, title, out_path, max_points=MAX_TSNE_POINTS):
    feat_model = extract_penultimate(model)
    feats, ys = [], []
    for x,y in ds:
        f = feat_model.predict(x, verbose=0)
        feats.append(f); ys.extend(np.argmax(y.numpy(), axis=1).tolist())
    X = np.vstack(feats); Y = np.array(ys)
    if len(X) > max_points:
        idx = np.random.RandomState(SEED).choice(len(X), size=max_points, replace=False)
        X = X[idx]; Y = Y[idx]
    X2 = TSNE(n_components=2, random_state=SEED, init="pca", perplexity=30).fit_transform(X)
    plt.figure(figsize=(6,5))
    for i,lab in enumerate(labels):
        sel = (Y==i); plt.scatter(X2[sel,0], X2[sel,1], s=8, alpha=.7, label=lab)
    plt.legend(); plt.title(title); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def run_dl_experiments(meta_images, transform, archs, out_root):
    df_t = meta_images[meta_images["transform"] == transform].reset_index(drop=True)
    if df_t.empty:
        print(f"[DL] No images for transform {transform}. Skipping.")
        return pd.DataFrame()

    ds_train, ds_val, ds_test, labels = prepare_tf_datasets(df_t)
    results = []
    for arch in archs:
        title = f"{arch}_{transform}"
        out_dir = os.path.join(out_root, title)
        ensure_dirs(out_dir)

        print(f"\n=== Training {arch} on {transform} ===")
        model, base, unfz, ft_lr = make_model(arch, n_classes=len(labels))
        ck = keras.callbacks.ModelCheckpoint(os.path.join(out_dir, "best.keras"),
                                             monitor="val_accuracy", save_best_only=True, mode="max")
        es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

        hist = model.fit(ds_train, validation_data=ds_val, epochs=TF_EPOCHS, callbacks=[ck, es], verbose=1)
        plot_history(hist, title+"_stage1", out_dir)

        model = fine_tune(model, base, unfreeze_from=unfz, lr=ft_lr)
        hist2 = model.fit(ds_train, validation_data=ds_val, epochs=TF_FT_EPOCHS, callbacks=[ck, es], verbose=1)
        plot_history(hist2, title+"_stage2", out_dir)

        y_true = np.argmax(np.vstack([y.numpy() for _,y in ds_test]), axis=1)
        y_pred = np.argmax(model.predict(ds_test, verbose=0), axis=1)
        acc = float((y_true==y_pred).mean())
        p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

        print("\nClassification report:")
        print(classification_report(y_true, y_pred, target_names=labels, digits=4))

        plot_confusion(y_true, y_pred, labels, f"{title} – Confusion", os.path.join(out_dir,"confusion.png"))
        run_tsne(model, ds_test, labels, f"{title} – t-SNE", os.path.join(out_dir,"tsne.png"))

        results.append({"method": f"DL_{arch}_{transform}", "transform": transform, "arch": arch,
                        "accuracy": acc, "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1)})
    return pd.DataFrame(results)

# ---------- CNN-LSTM on sequences ----------
def prepare_seq_dataset(df, target_size=IMG_SIZE, batch=BATCH, T=8):
    labels = sorted(df["label"].unique())
    label2id = {l:i for i,l in enumerate(labels)}
    df = df.copy(); df["y"] = df["label"].map(label2id)

    tr, te = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["y"])
    tr, va = train_test_split(tr, test_size=0.2, random_state=SEED, stratify=tr["y"])

    def load_img_as_seq(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, target_size)
        slices = tf.split(img, num_or_size_splits=T, axis=1)  # along width
        seq = tf.stack(slices, axis=0)  # (T, H, W/T, 3)
        seq = tf.cast(seq, tf.float32)/255.0
        return seq

    def make_ds(dd, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((dd["img"].tolist(), dd["y"].tolist()))
        ds = ds.map(lambda p,y: (load_img_as_seq(p), tf.one_hot(y, depth=len(labels))),
                    num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle: ds = ds.shuffle(1024, seed=SEED)
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    return make_ds(tr,True), make_ds(va), make_ds(te), labels

def make_cnn_lstm(seq_shape, n_classes):
    x_in = keras.Input(shape=seq_shape)  # (T,H,W,3)
    cnn = keras.Sequential([
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),
        layers.GlobalAveragePooling2D()
    ])
    x = layers.TimeDistributed(cnn)(x_in)   # (B,T,F)
    x = layers.LSTM(128)(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(x_in, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def run_cnn_lstm_experiment(meta_images, transform_for_seq, out_root, T=8):
    df = meta_images[meta_images["transform"] == transform_for_seq].reset_index(drop=True)
    if df.empty:
        print(f"[CNN-LSTM] No images for {transform_for_seq}. Skipping.")
        return pd.DataFrame()
    ds_train, ds_val, ds_test, labels = prepare_seq_dataset(df, T=T)
    seq_shape = (T, IMG_SIZE[0], IMG_SIZE[1]//T, 3)
    model = make_cnn_lstm(seq_shape, n_classes=len(labels))

    out_dir = os.path.join(out_root, f"cnn_lstm_{transform_for_seq}")
    ensure_dirs(out_dir)
    ck = keras.callbacks.ModelCheckpoint(os.path.join(out_dir,"best.keras"), monitor="val_accuracy",
                                         save_best_only=True, mode="max")
    es = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

    hist = model.fit(ds_train, validation_data=ds_val, epochs=20, callbacks=[ck, es], verbose=1)

    # plots
    from cp_dl import plot_history, plot_confusion  # safe self import for plotting
    plot_history(hist, f"cnn_lstm_{transform_for_seq}", out_dir)

    y_true = np.argmax(np.vstack([y.numpy() for _,y in ds_test]), axis=1)
    y_pred = np.argmax(model.predict(ds_test, verbose=0), axis=1)
    acc = float((y_true==y_pred).mean())
    p,r,f1,_ = precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    print("\nCNN-LSTM report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    plot_confusion(y_true, y_pred, labels, f"CNN-LSTM {transform_for_seq} – Confusion",
                   os.path.join(out_dir,"confusion.png"))

    # t-SNE on LSTM outputs
    from tensorflow.keras import Model
    feat_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    feats, ys = [], []
    for xb,yb in ds_test:
        f = feat_model.predict(xb, verbose=0)
        feats.append(f); ys.extend(np.argmax(yb.numpy(), axis=1).tolist())
    X = np.vstack(feats); Y = np.array(ys)
    if len(X) > MAX_TSNE_POINTS:
        idx = np.random.RandomState(SEED).choice(len(X), MAX_TSNE_POINTS, replace=False)
        X, Y = X[idx], Y[idx]
    X2 = TSNE(n_components=2, random_state=SEED).fit_transform(X)
    plt.figure(figsize=(6,5))
    for i,lab in enumerate(labels):
        sel = (Y==i); plt.scatter(X2[sel,0], X2[sel,1], s=8, alpha=.7, label=lab)
    plt.legend(); plt.title(f"CNN-LSTM {transform_for_seq} – t-SNE"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir,"tsne.png"), dpi=150); plt.close()

    return pd.DataFrame([{"method":f"DL_CNNLSTM_{transform_for_seq}","transform":transform_for_seq,
                          "arch":"CNNLSTM","accuracy":acc,"precision_macro":float(p),
                          "recall_macro":float(r),"f1_macro":float(f1)}])