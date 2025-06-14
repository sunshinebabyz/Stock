import sys
import os

if len(sys.argv) < 5:
    sys.exit(1)

symbol = sys.argv[1]
window_size = int(sys.argv[2])
epochs = int(sys.argv[3])
batch_size = int(sys.argv[4])
model_type = sys.argv[5].lower() if len(sys.argv) > 5 else 'lstm'

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', f'{symbol}.csv')

if not os.path.exists(file_path):
    sys.exit(1)

try:
    from ml import train_lstm_model_from_csv, train_gru_model_from_csv
except Exception as import_error:
    with open(f"train_{symbol}_error.txt", "w") as f:
        f.write(f"Lỗi import: {import_error}")
    sys.exit(1)

try:
    if model_type == 'gru':
        train_gru_model_from_csv(file_path, window_size, epochs, batch_size)
    else:
        train_lstm_model_from_csv(file_path, window_size, epochs, batch_size)
    print(f"✅ Training {model_type.upper()} model for {symbol} complete.")
    sys.exit(0)

except Exception as e:
    with open(f"train_{symbol}_error.txt", "w") as f:
        f.write(str(e))
    sys.exit(1)
