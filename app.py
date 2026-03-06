from flask import Flask, render_template, jsonify, request
import requests
import joblib
import json
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the REAL trained model
try:
    model = joblib.load('models/lightgbm_model.pkl')
    calibrator = joblib.load('models/calibrator.pkl')
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    MODEL_LOADED = True
    print("✅ Real model loaded successfully!")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠️ Could not load model: {e}")
    model = None
    calibrator = None
    feature_names = []

# Model configuration matching old demo structure
MODELS = {
    'lightgbm': {
        'name': 'LightGBM',
        'status': 'Trained on Real Data',
        'r2': None,  # Not applicable for classification
        'auc_roc': 0.9298,
        'rmse': None,
        'mae': None,
        'training_time': '~2 min',
        'inference_speed': '~50ms',
        'rank': '🏆 Production Model',
        'pros': [
            'Trained on 300K real impressions with actual user clicks',
            'AUC-ROC 0.93 (excellent discrimination)',
            'Well calibrated (2.05% predicted vs 1.75% actual CTR)',
            'Handles cold-start tiles via is_boost signal',
            'Adapts intra-day as tiles get impressions',
            'Timestamp-based lagging enables same-day learning'
        ],
        'cons': [
            'Only 7 days of training data (30+ days recommended)',
            'Predictions slightly optimistic (+0.3% CTR)',
            'Needs real-time CTR infrastructure for production',
            'Model should be retrained weekly'
        ],
        'best_for': 'Production deployment with real user data',
        'technical': 'LightGBM classifier with timestamp-based feature lagging, isotonic calibration, 43 features including user behavior and real-time tile CTR.',
        'personalization': 'Strong - user_total_sessions (#1), user_sessions_with_clicks (#4), user_type (#10)',
        'training_samples': '240,236 impressions',
        'test_samples': '60,060 impressions',
        'actual_ctr': '1.75%',
        'predicted_ctr': '2.05%',
        'total_clicks': '5,842',
        'training_period': 'Feb 27 - Mar 6, 2026',
        'features': 43,
        'top_features': [
            'user_total_sessions (2428)',
            'tile_ctr_lagged (2351)',
            'hour_of_day (1727)',
            'user_sessions_with_clicks (1485)',
            'tile_position (1385)'
        ]
    }
}

USER_PROFILES = {
    'new': {
        'name': 'New User',
        'strategy': 'First visit - show NCOs and popular bets',
        'sessions': 0,
        'preferred_bet_type': 'Single',
        'preferred_bookmaker': 'Sky Bet'
    },
    'casual': {
        'name': 'Casual Bettor',
        'strategy': 'Occasional bettor - balanced mix',
        'sessions': 5,
        'preferred_bet_type': 'Acca',
        'preferred_bookmaker': 'Sky Bet'
    },
    'power': {
        'name': 'Power User',
        'strategy': 'Frequent bettor - boost tiles and accas',
        'sessions': 50,
        'preferred_bet_type': 'Mega Acca',
        'preferred_bookmaker': 'Sky Bet'
    }
}

@app.route('/')
def index():
    """Landing page with live demo"""
    return render_template('index.html',
                         models=MODELS,
                         user_profiles=USER_PROFILES,
                         model_loaded=MODEL_LOADED)

@app.route('/api/fetch-tiles', methods=['GET', 'POST'])
def fetch_tiles():
    """Fetch live tiles from Checkd API"""
    try:
        response = requests.get('https://api-prod.checkd.media/discover/feed', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            tiles = data.get('tiles', [])
            
            # Add original position
            for i, tile in enumerate(tiles):
                tile['position'] = i + 1
                tile['original_position'] = i + 1
            
            return jsonify({
                'success': True,
                'tiles': tiles,
                'count': len(tiles)
            })
        else:
            return jsonify({
                'success': False,
                'error': f'API returned status {response.status_code}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rank-tiles', methods=['GET', 'POST'])
def rank_tiles():
    """Rank tiles using the REAL trained model"""
    try:
        data = request.json
        tiles = data.get('tiles', [])
        model_name = data.get('model', 'lightgbm')
        user_type = data.get('user_type', 'casual')
        
        # Get user profile
        user_profile = USER_PROFILES.get(user_type, USER_PROFILES['casual'])
        
        if not MODEL_LOADED:
            # Fallback: simple boost-based ranking
            scored_tiles = []
            for tile in tiles:
                # Extract keywords as dict
                kw = {}
                if 'keywords' in tile:
                    for k in tile['keywords']:
                        kw[k['name']] = k['value']
                
                bet_type = kw.get('bet_type', '')
                
                # Simple heuristic scoring
                if 'boost' in bet_type.lower():
                    score = 0.45
                elif tile.get('tile_type') == 'New Customer Offer':
                    score = 0.08
                else:
                    score = 0.02
                
                scored_tiles.append({
                    'tile': tile,
                    'score': score,
                    'original_position': tile.get('original_position', tile.get('position', 0))
                })
            
            # Sort by score
            scored_tiles.sort(key=lambda x: x['score'], reverse=True)
            
            # Add recommended position and change
            for i, item in enumerate(scored_tiles):
                item['recommended_position'] = i + 1
                item['change'] = item['original_position'] - item['recommended_position']
            
            return jsonify({
                'success': True,
                'tiles': scored_tiles,
                'model': MODELS['lightgbm'],
                'user_profile': user_profile,
                'note': 'Model not loaded, using heuristic'
            })
        
        # Use REAL model for scoring
        scored_tiles = []
        
        for tile in tiles:
            # Extract keywords as dict
            kw = {}
            if 'keywords' in tile:
                for k in tile['keywords']:
                    kw[k['name']] = k['value']
            
            bet_type = kw.get('bet_type', '')
            
            # Create feature dict (simplified - matching training features)
            features = {}
            
            # Numeric features
            features['tile_position'] = tile.get('original_position', tile.get('position', 1))
            features['is_boost'] = 1 if 'boost' in bet_type.lower() else 0
            features['is_nco'] = 1 if tile.get('tile_type') == 'New Customer Offer' else 0
            features['is_weekend'] = 1
            features['hour_of_day'] = 14
            features['day_of_week'] = 5
            features['tile_age_days'] = 0
            features['tile_ctr_lagged'] = 0.02  # Would come from real-time system in production
            
            # User features based on profile
            features['user_total_sessions'] = user_profile['sessions']
            features['user_sessions_with_clicks'] = int(user_profile['sessions'] * 0.6)
            
            # Add all feature columns (one-hot encoded will be 0 by default)
            for fname in feature_names:
                if fname not in features:
                    features[fname] = 0
            
            # Create DataFrame with correct column order
            df_row = pd.DataFrame([features])[feature_names]
            
            # Get prediction
            pred_uncal = model.predict_proba(df_row)[0][1]
            pred = calibrator.transform([pred_uncal])[0]
            
            scored_tiles.append({
                'tile': tile,
                'score': float(pred),
                'original_position': tile.get('original_position', tile.get('position', 0))
            })
        
        # Sort by score
        scored_tiles.sort(key=lambda x: x['score'], reverse=True)
        
        # Add recommended position and change
        for i, item in enumerate(scored_tiles):
            item['recommended_position'] = i + 1
            item['change'] = item['original_position'] - item['recommended_position']
        
        return jsonify({
            'success': True,
            'tiles': scored_tiles,
            'model': MODELS['lightgbm'],
            'user_profile': user_profile
        })
        
    except Exception as e:
        print(f"Error in rank_tiles: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)