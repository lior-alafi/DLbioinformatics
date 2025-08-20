from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr


def metrics(model_name,train_losses,val_loss,
            val_pearson,val_spearman):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # יצירת שם קובץ כולל כל ההיפרפרמטרים
    filename = f"training_metrics_{model_name.replace(' ','_')}_{timestamp}.png"

    # יצירת תרשימים
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # גרף 1: Train loss
    axs[0, 0].plot(train_losses, label="Train Loss", color="blue")
    axs[0, 0].set_title("Train Loss")
    axs[0, 0].set_ylabel("Avg Loss")
    axs[0, 0].legend()

    # גרף 2: Validation loss
    axs[0, 1].plot(val_loss, label="Val Loss", color="orange")
    axs[0, 1].set_title("Validation Loss")
    axs[0, 1].set_ylabel("Avg Loss")
    axs[0, 1].legend()

    # גרף 3: Pearson
    axs[1, 0].plot(val_pearson, label="Pearson", color="green")
    axs[1, 0].set_title("Pearson Correlation")
    axs[1, 0].set_ylabel("Pearson")
    axs[1, 0].legend()

    # גרף 4: Spearman
    axs[1, 1].plot(val_spearman, label="Spearman", color="red")
    axs[1, 1].set_title("Spearman Correlation")
    axs[1, 1].set_ylabel("Spearman")
    axs[1, 1].legend()

    # התאמה סופית
    plt.suptitle(f"Training Metrics - {model_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename.replace(':','_'))
    plt.close()

    print(f"Saved training metrics to: {filename}")

def pearson_evaluation(y_orig, y_hats, bad_pearson_count, epoch, scaler=None, use_inverse=False, std_eps=1e-6):
    # y_orig / y_hats: רשימות או np arrays של סקאלרים
    y_orig = np.asarray(y_orig, dtype=np.float64).reshape(-1, 1)
    y_hats = np.asarray(y_hats, dtype=np.float64).reshape(-1, 1)

    # אופציונלי בלבד: חזרה ל-unscaled (לא נחוץ לקורלציה)
    if use_inverse and scaler is not None:
        y_orig = scaler.inverse_transform(y_orig)
        y_hats = scaler.inverse_transform(y_hats)

    y_true = y_orig.ravel()
    y_pred = y_hats.ravel()

    # בדיקות בטיחות
    if not np.all(np.isfinite(y_pred)) or not np.all(np.isfinite(y_true)):
        print(f"⚠️ epoch {epoch+1}: non-finite values detected; skipping Pearson.")
        bad_pearson_count += 1
        return bad_pearson_count, 0.0, 0.0

    std_y_hat = np.std(y_pred)
    std_y_true = np.std(y_true)

    # דיאגנוסטיקה: קריסת הווריאנס של התחזיות -> Pearson≈0
    print(f"[diag e{epoch+1}] var(y)={np.var(y_true):.4f} var(ŷ)={np.var(y_pred):.4f} mean(ŷ)={np.mean(y_pred):.4f}")

    if std_y_hat < std_eps or std_y_true < std_eps:
        print(f"⚠️ epoch {epoch+1}: near-zero std (ŷ: {std_y_hat:.6f}, y: {std_y_true:.6f}) → set Pearson=0.")
        bad_pearson_count += 1
        return bad_pearson_count, 0.0, 0.0

    # חישוב הקורלציות
    p_val = pearsonr(y_true, y_pred)[0]
    s_val = spearmanr(y_true, y_pred)[0]

    # טיפול במקרה נדיר של NaN
    if not np.isfinite(p_val):
        p_val = 0.0
        bad_pearson_count += 1
    else:
        bad_pearson_count = 0

    if not np.isfinite(s_val):
        s_val = 0.0

    return bad_pearson_count, float(p_val), float(s_val)


