from datetime import datetime

from matplotlib import pyplot as plt


def metrics(train_losses,val_loss,
            val_pearson,val_spearman,
            HIDDEN_SIZE,LSTM_LAYER,LR,EPOCHS,EMBED_DIM):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    # יצירת שם קובץ כולל כל ההיפרפרמטרים
    filename = f"training_metrics_LSTM{HIDDEN_SIZE}_l{LSTM_LAYER}_lr{LR}_e{EPOCHS}_embd_{EMBED_DIM}_{timestamp}.png"

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
    plt.suptitle(f"Training Metrics - LSTM({HIDDEN_SIZE}, layers={LSTM_LAYER}, LR={LR}, Epochs={EPOCHS})", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close()

    print(f"Saved training metrics to: {filename}")