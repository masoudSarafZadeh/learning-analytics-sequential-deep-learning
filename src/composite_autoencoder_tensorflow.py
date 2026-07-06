import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import joblib

BASE_PATH = "path/to/your/downloaded/oulad/anonymisedData"

# ==========================================
# 1. Data Loading and Merging
# ==========================================
df_ass = pd.read_csv(f"{BASE_PATH}/assessments.csv")
df_stuass = pd.read_csv(f"{BASE_PATH}/studentAssessment.csv")
ass_stuass = df_stuass.merge(df_ass, on="id_assessment", how="outer")

# Sort and encode
aca = ass_stuass.sort_values(by=['id_student', 'id_assessment'], ascending=True)
label_cmcp = joblib.load('processed_data/label_encoder.joblib')
aca['code_module - code_presentation'] = label_cmcp.transform(aca['code_module - code_presentation'])

# Create unique student IDs based on module presentation
aca['id_student'] = (aca['id_student'] + (aca['code_module - code_presentation'] / 100)) * 100
aca.drop(['code_module - code_presentation'], axis=1, inplace=True)

# Filter out bad weights
test_weights = aca[aca['weight'] == 0]
mask = test_weights.groupby("id_student").size() > 1
mask = mask.reset_index()
true_ids = mask.id_student.unique().tolist()
aca = aca[~aca['id_student'].isin(true_ids)].copy()

# Feature Engineering
aca['pure_score'] = aca['score'] * aca['weight'] / 100
aca.drop(['id_assessment', 'is_banked', 'date_submitted', 'weight'], axis=1, inplace=True, errors='ignore')
aca = aca.sort_values(by=['id_student', 'date'])

# ==========================================
# 2. Train / Val / Test Split BEFORE Scaling
# ==========================================
df_all = pd.read_csv("processed_data/df_all.csv")
unique_students = df_all['id_student'].unique()

train_val_ids, test_ids = train_test_split(unique_students, test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_val_ids, test_size=0.18, random_state=42)

def process_assessment_type(df, ass_type, train_ids, val_ids, test_ids, scaler=None):
    df_type = df[df["assessment_type"] == ass_type].copy()
    prefix = f"_{ass_type}"
    df_type = df_type.rename(columns={"score": f"score{prefix}", "pure_score": f"pure_score{prefix}"})
    df_type = df_type.sort_values(by=['id_student', 'date'])
    df_type.drop(['assessment_type', 'date'], axis=1, inplace=True)
 
    train_df = df_type[df_type['id_student'].isin(train_ids)].copy()
    val_df = df_type[df_type['id_student'].isin(val_ids)].copy()
    test_df = df_type[df_type['id_student'].isin(test_ids)].copy()
    
    cols_to_scale = [f"score{prefix}", f"pure_score{prefix}"]
    
    if scaler is None:
        scaler = StandardScaler()
        train_df[cols_to_scale] = scaler.fit_transform(train_df[cols_to_scale])
    else:
        train_df[cols_to_scale] = scaler.transform(train_df[cols_to_scale])
        
    val_df[cols_to_scale] = scaler.transform(val_df[cols_to_scale])
    test_df[cols_to_scale] = scaler.transform(test_df[cols_to_scale])
    
    # Recombine temporarily for pivoting
    combined_scaled = pd.concat([train_df, val_df, test_df])
    combined_scaled['seq'] = combined_scaled.groupby('id_student').cumcount() + 1
    
    wide_df = combined_scaled.pivot(index='id_student', columns='seq', values=[f'pure_score{prefix}', f'score{prefix}'])
    wide_df.columns = [f"{col}_{seq}" for col, seq in wide_df.columns]
    wide_df = wide_df.reset_index().fillna(0.0) # Used 0.0 for scaled missing values instead of -10.0
    
    return wide_df, scaler

wide_TMA, scaler_TMA = process_assessment_type(aca, "TMA", train_ids, val_ids, test_ids)
wide_CMA, scaler_CMA = process_assessment_type(aca, "CMA", train_ids, val_ids, test_ids)

ass_result = wide_TMA.merge(wide_CMA, how='right', on='id_student')
ass_result = ass_result.merge(df_all, how='right', on='id_student')

feature_cols = [
    'pure_score_TMA_1', 'score_TMA_1', 'pure_score_CMA_1', 'score_CMA_1',
    'pure_score_TMA_2', 'score_TMA_2', 'pure_score_CMA_2', 'score_CMA_2',
    'pure_score_TMA_3', 'score_TMA_3', 'pure_score_CMA_3', 'score_CMA_3',
    'pure_score_TMA_4', 'score_TMA_4', 'pure_score_CMA_4', 'score_CMA_4',
    'pure_score_TMA_5', 'score_TMA_5', 'pure_score_CMA_5', 'score_CMA_5',
    'pure_score_TMA_6', 'score_TMA_6', 'pure_score_CMA_6', 'score_CMA_6'
]

train_data = ass_result[ass_result['id_student'].isin(train_ids)]
val_data = ass_result[ass_result['id_student'].isin(val_ids)]
test_data = ass_result[ass_result['id_student'].isin(test_ids)]

# Reshape to (Samples, Timesteps, Features)
def prep_keras_data(df, feature_cols):
    X = df[feature_cols].values.astype('float32')
    y = df['final_result'].values.astype('float32')
    X_reshaped = np.reshape(X, (X.shape[0], 6, 4))
    return X_reshaped, y

trainX, y_train = prep_keras_data(train_data, feature_cols)
valX, y_val = prep_keras_data(val_data, feature_cols)
testX, y_test = prep_keras_data(test_data, feature_cols)

# ==========================================
# 4. Composite Autoencoder Architecture
# ==========================================
timesteps = 6
input_dim = 4

inputs = Input(shape=(timesteps, input_dim), name="input_6")

# Encoder
encoded = LSTM(250, activation='relu', return_sequences=True)(inputs)
encoded = LSTM(200, activation='relu', return_sequences=True)(encoded)
encoded = LSTM(100, activation='relu', return_sequences=True)(encoded)

# The Latent Representation (Bottleneck)
bottleneck = LSTM(2, activation='relu', return_sequences=False, name="latent_representation")(encoded)

# Branch 1: Reconstruction Decoder
decoded = RepeatVector(timesteps)(bottleneck)
decoded = LSTM(100, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(150, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(200, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(150, activation='relu', return_sequences=True)(decoded)
reconstruction_output = TimeDistributed(Dense(input_dim), name="reconstruction")(decoded)

# Branch 2: Binary Classification Head
clf = Dense(64, activation='relu')(bottleneck)
clf = Dense(128, activation='relu')(clf)
classification_output = Dense(1, activation='sigmoid', name="classification")(clf)

# Compile Composite Model
composite_model = Model(inputs=inputs, outputs=[reconstruction_output, classification_output])
composite_model.compile(
    optimizer=Adam(learning_rate=0.0001, clipnorm=1.0), 
    loss={'reconstruction': 'mse', 'classification': 'binary_crossentropy'},
    loss_weights={'reconstruction': 1.0, 'classification': 1.0}
)

print(composite_model.summary())

# ==========================================
# 5. Training
# ==========================================
history = composite_model.fit(
    trainX, 
    {'reconstruction': trainX, 'classification': y_train},
    validation_data=(valX, {'reconstruction': valX, 'classification': y_val}),
    epochs=100, 
    batch_size=64,
    verbose=1
)

# ==========================================
# 6. Extracting the Latent Representation 
# ==========================================
# Now we extract the sub-model that outputs the 2-neuron embedding
encoder_model = Model(inputs=composite_model.input, outputs=composite_model.get_layer('latent_representation').output)

# These embeddings are what we will pass to the Metamodel later
encoded_features_train = encoder_model.predict(trainX)
encoded_features_test = encoder_model.predict(testX)

# ==========================================
# 7. Evaluation
# ==========================================
y_pred_probs = composite_model.predict(testX)[1] # Index 1 gets the classification output
threshold = (np.max(y_pred_probs) + np.min(y_pred_probs)) / 2
y_pred_classes = (y_pred_probs > threshold).astype(int)

cm = confusion_matrix(y_test, y_pred_classes)
tn, fp, fn, tp = cm.ravel()

print(f"Accuracy: {accuracy_score(y_test, y_pred_classes):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_classes):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_classes):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_classes):.4f}")
print(f"Specificity: {(tn / (tn + fp)):.4f}")
