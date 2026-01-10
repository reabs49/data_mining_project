from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
app = Flask(__name__)

# Charger le modèle et le scaler
dossier_actuel = os.path.dirname(os.path.abspath(__file__))
chemin_modele = os.path.join(dossier_actuel, 'random_forest_optimized.pkl')
chemin_scaler = os.path.join(dossier_actuel, 'scaler.pkl')
model = joblib.load(chemin_modele)
scaler = joblib.load(chemin_scaler)
print("✅ Modèle et scaler chargés avec succès!")

def get_recommendations(risk_level):
    """Return safety recommendations based on risk level"""
    recommendations = {
        "CRITIQUE": [
            "🚨 Évacuer immédiatement la zone forestière",
            "📞 Contacter les services d'urgence et forestiers",
            "🚫 Interdire tout accès à la zone",
            "💧 Déployer les équipes de prévention si disponibles"
        ],
        "ÉLEVÉ": [
            "⚠️ Préparer un plan d'évacuation d'urgence",
            "👁️ Surveiller étroitement l'évolution des conditions",
            "🔥 Avoir l'équipement anti-incendie prêt",
            "📡 Maintenir une communication constante avec les autorités"
        ],
        "MODÉRÉ": [
            "👀 Rester vigilant et surveiller les conditions",
            "🚫 Éviter les activités à haut risque (feux, étincelles)",
            "🛡️ Maintenir l'équipement de sécurité accessible",
            "📊 Effectuer des mesures régulières des paramètres"
        ],
        "FAIBLE": [
            "✅ Continuer la surveillance standard",
            "🔍 Respecter les mesures de prévention de base",
            "📝 Signaler tout changement inhabituel",
            "🌲 Maintenir les zones coupe-feu propres"
        ]
    }
    return recommendations.get(risk_level, [])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Préparer les 20 features dans le bon ordre
        features = np.array([[
            data['COARSE'],
            data['CLAY'],
            data['TEXTURE_USDA'],
            data['CEC_CLAY'],
            data['TEB'],
            data['ALUM_SAT'],
            data['ESP'],
            data['TCARBON_EQ'],
            data['GYPSUM'],
            data['ELEC_COND'],
            data['elevation'],
            data['tmin_cool_wet'],
            data['tmax_hot_dry'],
            data['tmax_spring_transition'],
            data['prec_hot_dry'],
            data['prec_cool_wet'],
            data['prec_spring_transition'],
            data['TEXTURE_SOTER_C'],
            data['TEXTURE_SOTER_F'],
            data['TEXTURE_SOTER_M']
        ]])
        
        # Normaliser avec le scaler
        features_scaled = scaler.transform(features)
        
        # Prédiction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        # Probabilité de la classe "fire" (classe 1)
        risk_score = probability[1] * 100
        
        # Déterminer le niveau de risque
        if risk_score > 70:
            risk_level = "CRITIQUE"
            risk_color = "#ff3b3b"
            message = "⚠️ DANGER IMMÉDIAT - Risque d'incendie extrêmement élevé détecté. Conditions critiques pour un départ de feu."
        elif risk_score > 50:
            risk_level = "ÉLEVÉ"
            risk_color = "#ff8c00"
            message = "🔥 ATTENTION - Conditions très favorables aux incendies. Surveillance renforcée nécessaire."
        elif risk_score > 30:
            risk_level = "MODÉRÉ"
            risk_color = "#ffd700"
            message = "⚡ VIGILANCE - Conditions moyennement propices aux incendies. Rester attentif aux évolutions."
        else:
            risk_level = "FAIBLE"
            risk_color = "#4caf50"
            message = "✅ CONDITIONS NORMALES - Risque d'incendie faible. Surveillance standard recommandée."
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'risk_color': risk_color,
            'message': message,
            'recommendations': get_recommendations(risk_level),
            'probability_no_fire': round(probability[0] * 100, 2),
            'probability_fire': round(probability[1] * 100, 2)
        })
        
    except KeyError as e:
        return jsonify({
            'success': False, 
            'error': f'Feature manquante: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False, 
            'error': f'Erreur lors de la prédiction: {str(e)}'
        }), 500

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔥 ForestGuard AI - Système de Détection d'Incendies")
    print("="*70)
    print("✅ Serveur démarré sur http://127.0.0.1:5000")
    print("="*70 + "\n")
    app.run(debug=True)