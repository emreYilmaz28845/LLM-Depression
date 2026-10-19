import numpy as np
from tqdm import tqdm
import os
import sys
import subprocess
import tempfile
import pandas as pd
# ======================
# KULLANICI AYARLARI
# ======================
CSV_PATH = "/home/utku/projects/jad/nongeriatri_metadata_negative/metadata_negative_samples.csv"
OUT_CSV = "/home/utku/projects/jad/nongeriatri_metadata_negative/metadata_with_mfcc_nong"

SMILE_BIN = "/home/utku/opensmile/build/progsrc/smilextract/SMILExtract"
CONF_TEMPLATE_FILE = "/home/utku/emo_depression/xgboost/mffcc_config/avec2013_3.conf"

EXPAND_COLUMNS = False
FALLBACK_DIM = 384
FFMPEG_BIN = "ffmpeg"

# ======================
# === Yardımcı Fonksiyonlar
# ======================

def create_dynamic_conf(original_conf_path, input_file, output_file):
    """
    Orijinal config dosyasını okur, yer tutucuları doldurur ve geçici dosya oluşturur.
    """
    try:
        with open(original_conf_path, "r") as f:
            original_config = f.read()
    except FileNotFoundError:
        raise RuntimeError(f"Orijinal openSMILE config dosyası bulunamadı: {original_conf_path}")

    # Yer tutucuları doldur
    modified_config = original_config.replace("__INPUT_FILE_PLACEHOLDER__", input_file)
    modified_config = modified_config.replace("__OUTPUT_CSV_PLACEHOLDER__", output_file)

    # Geçici dosyayı oluştur
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".conf") as temp_conf:
        temp_conf.write(modified_config)
        return temp_conf.name

def parse_opensmile_output_file(path):
    """openSMILE çıktısını oku ve son satırı float array olarak döndür."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise ValueError("openSMILE çıktısı yok veya boş.")
    try:
        df = pd.read_csv(path, comment='@', sep=';|,', engine='python')
        row = df.iloc[-1].copy()
        row = pd.to_numeric(row, errors="coerce").dropna()
        if row.empty:
            raise ValueError("Sayısal değer bulunamadı.")
        return row.values.astype(float)
    except Exception as e:
        raise ValueError(f"Çıktı dosyası okunurken hata oluştu: {e}")

def run_smil_extract(conf_path):
    """Geçici config dosyasıyla openSMILE çalıştır."""
    command = [
        SMILE_BIN,
        "-C", conf_path,
        "-loglevel", "1"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"openSMILE hata verdi:\n{result.stderr.strip()[:300]}")


def ffmpeg_convert_to_pcm16k_mono(in_wav, out_wav):
    """WAV dosyasını 16kHz mono PCM formatına çevir."""
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", in_wav,
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        out_wav
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)


def extract_mfcc_features_from_file(wav_path):
    """Bir ses dosyasından MFCC çıkar ve array olarak döndür."""
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"WAV bulunamadı: {wav_path}")

    with tempfile.TemporaryDirectory() as td:
        tmp_out = os.path.join(td, "mfcc.csv")
        temp_conf_path = None
        try:
            temp_conf_path = create_dynamic_conf(CONF_TEMPLATE_FILE, wav_path, tmp_out)
            run_smil_extract(temp_conf_path)
            return parse_opensmile_output_file(tmp_out)
        except Exception as e1:
            try:
                print(f"Uyarı: İlk deneme başarısız. ffmpeg ile dönüştürülüyor...")
                conv_wav = os.path.join(td, "conv.wav")
                ffmpeg_convert_to_pcm16k_mono(wav_path, conv_wav)
                temp_conf_path = create_dynamic_conf(CONF_TEMPLATE_FILE, conv_wav, tmp_out)
                run_smil_extract(temp_conf_path)
                return parse_opensmile_output_file(tmp_out)
            except Exception as e2:
                raise RuntimeError(
                    f"MFCC çıkartılamadı: {wav_path}\n"
                    f"1) İlk deneme: {str(e1)[:300]}\n"
                    f"2) ffmpeg sonrası: {str(e2)[:300]}"
                )
        finally:
            if temp_conf_path and os.path.exists(temp_conf_path):
                os.remove(temp_conf_path)

# ======================
# === Ana Çalışma
# ======================
def main():
    if not os.path.isfile(SMILE_BIN):
        sys.exit(f"SMILExtract bulunamadı: {SMILE_BIN}")
    if not os.path.isfile(CONF_TEMPLATE_FILE):
        sys.exit(f"Config dosyası bulunamadı: {CONF_TEMPLATE_FILE}")
    if not os.path.isfile(CSV_PATH):
        sys.exit(f"Girdi CSV dosyası bulunamadı: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if "file_name" not in df.columns:
        sys.exit("CSV'de 'file_name' sütunu yok.")

    feature_len = None
    feature_names = None
    vec_strings = []
    mat_rows = []

    for wav in tqdm(df["file_name"], desc="Extracting MFCC"):
        try:
            arr = extract_mfcc_features_from_file(wav)

            # İlk 2 değeri at
            if len(arr) > 2:
                arr = arr[2:]

            
            if feature_len is None:
                feature_len = len(arr)
                feature_names = [f"mfcc_{i}" for i in range(feature_len)]
            
            if len(arr) != feature_len:
                arr = np.resize(arr, feature_len)
            
            if EXPAND_COLUMNS:
                mat_rows.append(arr)
            else:
                vec_strings.append(",".join(f"{v:.6f}" if np.isfinite(v) else "nan" for v in arr))
        
        except Exception as e:
            print(f"❌ Hata: {wav} -> {e}")
            fill = feature_len if feature_len else FALLBACK_DIM
            if EXPAND_COLUMNS:
                mat_rows.append(np.full(fill, np.nan))
            else:
                vec_strings.append(",".join(["nan"] * fill))

    if EXPAND_COLUMNS:
        mat = np.vstack(mat_rows)
        if feature_names is None:
            feature_names = [f"mfcc_{i}" for i in range(mat.shape[1])]
        out_df = pd.concat([df.reset_index(drop=True),
                            pd.DataFrame(mat, columns=feature_names)], axis=1)
    else:
        out_df = df.copy()
        out_df["features"] = vec_strings

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"✅ Kaydedildi: {OUT_CSV}")

if __name__ == "__main__":
    main()