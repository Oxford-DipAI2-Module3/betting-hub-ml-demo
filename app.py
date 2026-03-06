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

# Real model performance from training
REAL_MODEL_PERFORMANCE = {
    'name': 'LightGBM (Production)',
    'status': 'Trained on Real Data',
    'auc_roc': 0.9298,
    'actual_ctr': 0.0175,
    'predicted_ctr': 0.0205,
    'training_samples': 240236,
    'test_samples': 60060,
    'training_period': 'Feb 27 - Mar 6, 2026 (7 days)',
    'total_clicks': 5842,
    'features': 43,
    'top_features': [
        {'name': 'user_total_sessions', 'importance': 2428},
        {'name': 'tile_ctr_lagged', 'importance': 2351},
        {'name': 'hour_of_day', 'importance': 1727},
        {'name': 'user_sessions_with_clicks', 'importance': 1485},
        {'name': 'tile_position', 'importance': 1385},
    ],
    'key_insights': [
        'Timestamp-based lagging enables intra-day learning',
        '97.3% of impressions have 50+ prior data points',
        'Boost tiles (42.81% CTR) vs Regular (1.01% CTR)',
        'Model adapts within same day (0% → 92% CTR by impression 50)',
        'is_boost feature handles cold-start (rank #13)'
    ],
    'pros': [
        'Trained on 300K real impressions with actual user clicks',
        'AUC-ROC 0.93 (excellent discrimination)',
        'Well calibrated (2.05% predicted vs 1.75% actual)',
        'Handles cold-start tiles via is_boost signal',
        'Adapts intra-day as tiles get impressions'
    ],
    'cons': [
        'Only 7 days of training data (30+ days recommended)',
        'Predictions slightly optimistic (+0.3% CTR)',
        'Needs real-time CTR infrastructure for production',
        'Model should be retrained weekly'
    ],
    'next_steps': [
        'Collect 30 days of data (by April 6)',
        'Build real-time CTR tracking system',
        'Deploy A/B test (5% of users)',
        'Monitor predicted vs actual CTR daily'
    ]
}

@app.route('/')
def index():
    """Landing page with model overview"""
    # Format data to match old template structure
    models_dict = {
        'lightgbm': REAL_MODEL_PERFORMANCE
    }
    
    user_profiles_dict = {
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
    
    return render_template('index.html',
                         models=models_dict,
                         user_profiles=user_profiles_dict,
                         model_loaded=MODEL_LOADED)

@app.route('/live-demo')
def live_demo():
    """Interactive demo with real Checkd tiles"""
    return render_template('live_demo.html')

@app.route('/api/fetch-tiles', methods=['POST'])
def fetch_tiles():
    """Fetch live tiles from Checkd API"""
    try:
        response = requests.get('https://api-prod.checkd.media/discover/feed', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            tiles = data.get('tiles', [])
            
            # Add original position
            for i, tile in enumerate(tiles):
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

@app.route('/api/rank-tiles', methods=['POST'])
def rank_tiles():
    """Rank tiles using the REAL trained model"""
    try:
        data = request.json
        tiles = data.get('tiles', [])
        user_type = data.get('userType', 'casual')
        
        if not MODEL_LOADED:
            # Fallback: simple boost-based ranking
            for tile in tiles:
                keywords = tile.get('keywords', {})
                bet_type = keywords.get('bet_type', '')
                
                # Simple heuristic scoring
                if 'boost' in bet_type.lower():
                    tile['ml_score'] = 0.45
                elif tile.get('tile_type') == 'New Customer Offer':
                    tile['ml_score'] = 0.08
                else:
                    tile['ml_score'] = 0.02
            
            # Sort by score
            tiles.sort(key=lambda x: x.get('ml_score', 0), reverse=True)
            
            return jsonify({
                'success': True,
                'tiles': tiles,
                'model_used': 'fallback_heuristic',
                'note': 'Real model not loaded, using simple heuristic'
            })
        
        # Use REAL model for scoring
        tile_features = []
        
        for tile in tiles:
            keywords = tile.get('keywords', {})
            
            # Extract features (simplified - in production you'd have all 43)
            features = {
                'tile_position': tile.get('original_position', 1),
                'is_boost': 1 if 'boost' in keywords.get('bet_type', '').lower() else 0,
                'is_nco': 1 if tile.get('tile_type') == 'New Customer Offer' else 0,
                'is_weekend': 1,  # Placeholder
                'hour_of_day': 14,  # Placeholder (2pm)
                'day_of_week': 5,  # Placeholder (Friday)
                'tile_age_days': 0,  # Placeholder (new tile)
                'user_total_sessions': 15 if user_type == 'power' else (5 if user_type == 'casual' else 1),
                'user_sessions_with_clicks': 8 if user_type == 'power' else (2 if user_type == 'casual' else 0),
                'tile_ctr_lagged': 0.02,  # Placeholder (would come from real-time system)
            }
            
            tile_features.append(features)
        
        # Create DataFrame with features
        df = pd.DataFrame(tile_features)
        
        # Add one-hot encoded features (simplified)
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Ensure column order matches training
        df = df[feature_names]
        
        # Get predictions
        predictions_uncal = model.predict_proba(df)[:, 1]
        predictions = calibrator.transform(predictions_uncal)
        
        # Add scores to tiles
        for i, tile in enumerate(tiles):
            tile['ml_score'] = float(predictions[i])
        
        # Sort by ML score
        tiles.sort(key=lambda x: x.get('ml_score', 0), reverse=True)
        
        return jsonify({
            'success': True,
            'tiles': tiles,
            'model_used': 'lightgbm_real_data',
            'note': 'Using production model trained on real CTR data'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/performance')
def performance():
    """Model performance metrics page"""
    return render_template('performance.html', metrics=REAL_MODEL_PERFORMANCE)

@app.route('/technical')
def technical():
    """Technical details and implementation"""
    return render_template('technical.html', model_info=REAL_MODEL_PERFORMANCE)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)