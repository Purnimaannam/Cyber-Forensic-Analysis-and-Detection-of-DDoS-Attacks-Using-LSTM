import os
import numpy as np
import joblib
from datetime import datetime, timedelta
from collections import Counter

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.cache import never_cache
from django.db.models import Count, Q, Avg
from tensorflow.keras.models import load_model

from .models import Prediction

MODEL_DIR = os.path.join(settings.BASE_DIR, "ml_models")

model = load_model(os.path.join(MODEL_DIR, "ddos_lstm_model.h5"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

activity_encoder = joblib.load(os.path.join(MODEL_DIR, "activity_encoder.pkl"))
action_encoder   = joblib.load(os.path.join(MODEL_DIR, "action_encoder.pkl"))
anomaly_encoder  = joblib.load(os.path.join(MODEL_DIR, "anomaly_encoder.pkl"))

def index(request):
    result = None
    confidence = None

    activities = list(activity_encoder.classes_)
    actions    = list(action_encoder.classes_)
    anomalies  = list(anomaly_encoder.classes_)

    if request.method == "POST":
        now = datetime.now()

        activity = activity_encoder.transform([request.POST['activity']])[0]
        action   = action_encoder.transform([request.POST['action']])[0]
        anomaly  = anomaly_encoder.transform([request.POST['anomaly']])[0]

        login_attempts = float(request.POST['login_attempts'])
        file_size      = float(request.POST['file_size'])

        hour = now.hour
        day  = now.day

        row = [[
            activity,
            action,
            login_attempts,
            file_size,
            anomaly,
            hour,
            day
        ]]

        row_scaled = scaler.transform(row)
        row_scaled = row_scaled.reshape(1, 1, row_scaled.shape[1])

        # Get model prediction
        prediction_prob = model.predict(row_scaled, verbose=0)
        confidence = float(prediction_prob[0][0]) * 100
        confidence = round(max(0, min(100, confidence)), 2)

        # Use dynamic threshold - classify as suspicious if probability > 0.3
        if prediction_prob[0][0] > 0.3:
            result = "🚨 DDoS / Suspicious Activity"
            prediction_result = "Attack"
        else:
            result = "✅ Normal Traffic"
            prediction_result = "Normal"

        # Save prediction to database
        Prediction.objects.create(
            activity_type=request.POST['activity'],
            action=request.POST['action'],
            login_attempts=login_attempts,
            file_size=file_size,
            anomaly_type=request.POST['anomaly'],
            prediction_result=prediction_result,
            confidence_score=confidence,
            hour=hour,
            day=day
        )

    return render(request, "index.html", {
        "result": result,
        "confidence": confidence,
        "activities": activities,
        "actions": actions,
        "anomalies": anomalies
    })


def analysis(request):
    """AI Analysis page with threat statistics and insights"""
    
    # Fetch all predictions
    all_predictions = Prediction.objects.all()
    
    # =================== BASIC STATISTICS ===================
    total_predictions = all_predictions.count()
    
    if total_predictions == 0:
        return render(request, "analysis.html", {
            "error": "No prediction data available yet. Submit some predictions to see analysis.",
            "total": 0
        })
    
    attack_count = all_predictions.filter(prediction_result="Attack").count()
    normal_count = all_predictions.filter(prediction_result="Normal").count()
    
    attack_rate = (attack_count / total_predictions * 100) if total_predictions > 0 else 0
    detection_rate = attack_rate
    
    # =================== CONFIDENCE ANALYSIS ===================
    avg_confidence = all_predictions.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
    avg_confidence = round(avg_confidence, 2)
    
    attack_predictions = all_predictions.filter(prediction_result="Attack")
    avg_attack_confidence = attack_predictions.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
    avg_attack_confidence = round(avg_attack_confidence, 2)
    
    normal_predictions = all_predictions.filter(prediction_result="Normal")
    avg_normal_confidence = normal_predictions.aggregate(Avg('confidence_score'))['confidence_score__avg'] or 0
    avg_normal_confidence = round(avg_normal_confidence, 2)
    
    # =================== THREAT PATTERNS ===================
    most_common_attacks = (
        attack_predictions
        .values('activity_type')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )
    
    most_common_actions = (
        attack_predictions
        .values('action')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )
    
    most_common_anomalies = (
        attack_predictions
        .values('anomaly_type')
        .annotate(count=Count('id'))
        .order_by('-count')[:5]
    )
    
    # =================== TIME-BASED ANALYSIS ===================
    hourly_attacks = (
        attack_predictions
        .values('hour')
        .annotate(count=Count('id'))
        .order_by('hour')
    )
    
    daily_attacks = (
        attack_predictions
        .values('day')
        .annotate(count=Count('id'))
        .order_by('day')
    )
    
    # =================== RISK METRICS ===================
    high_confidence_attacks = attack_predictions.filter(confidence_score__gte=70).count()
    medium_confidence_attacks = attack_predictions.filter(confidence_score__gte=50, confidence_score__lt=70).count()
    low_confidence_attacks = attack_predictions.filter(confidence_score__lt=50).count()
    
    # =================== AI ANALYSIS TEXT ===================
    analysis_text = generate_ai_analysis(
        total_predictions=total_predictions,
        attack_count=attack_count,
        normal_count=normal_count,
        attack_rate=attack_rate,
        avg_attack_confidence=avg_attack_confidence,
        most_common_attacks=most_common_attacks,
        most_common_actions=most_common_actions,
        high_confidence_attacks=high_confidence_attacks,
        medium_confidence_attacks=medium_confidence_attacks,
        low_confidence_attacks=low_confidence_attacks,
        hourly_attacks=hourly_attacks,
        daily_attacks=daily_attacks
    )
    
    # =================== LOGIN ATTEMPTS ANALYSIS ===================
    avg_login_attacks = attack_predictions.aggregate(Avg('login_attempts'))['login_attempts__avg'] or 0
    avg_login_normal = normal_predictions.aggregate(Avg('login_attempts'))['login_attempts__avg'] or 0
    avg_login_attacks = round(avg_login_attacks, 2)
    avg_login_normal = round(avg_login_normal, 2)
    
    # =================== FILE SIZE ANALYSIS ===================
    avg_file_attacks = attack_predictions.aggregate(Avg('file_size'))['file_size__avg'] or 0
    avg_file_normal = normal_predictions.aggregate(Avg('file_size'))['file_size__avg'] or 0
    avg_file_attacks = round(avg_file_attacks, 2)
    avg_file_normal = round(avg_file_normal, 2)
    
    context = {
        "total_predictions": total_predictions,
        "attack_count": attack_count,
        "normal_count": normal_count,
        "attack_rate": round(attack_rate, 2),
        "avg_confidence": avg_confidence,
        "avg_attack_confidence": avg_attack_confidence,
        "avg_normal_confidence": avg_normal_confidence,
        "most_common_attacks": list(most_common_attacks),
        "most_common_actions": list(most_common_actions),
        "most_common_anomalies": list(most_common_anomalies),
        "hourly_attacks": list(hourly_attacks),
        "daily_attacks": list(daily_attacks),
        "high_confidence_attacks": high_confidence_attacks,
        "medium_confidence_attacks": medium_confidence_attacks,
        "low_confidence_attacks": low_confidence_attacks,
        "avg_login_attacks": avg_login_attacks,
        "avg_login_normal": avg_login_normal,
        "avg_file_attacks": avg_file_attacks,
        "avg_file_normal": avg_file_normal,
        "analysis_text": analysis_text,
    }
    
    return render(request, "analysis.html", context)


def generate_ai_analysis(**metrics):
    """Generate comprehensive AI-driven analysis text"""
    
    total = metrics['total_predictions']
    attacks = metrics['attack_count']
    attack_rate = metrics['attack_rate']
    avg_confidence = metrics['avg_attack_confidence']
    most_common = metrics['most_common_attacks']
    high_conf = metrics['high_confidence_attacks']
    medium_conf = metrics['medium_confidence_attacks']
    low_conf = metrics['low_confidence_attacks']
    
    analysis = []
    
    # ========== THREAT OVERVIEW ==========
    analysis.append("THREAT OVERVIEW")
    if attack_rate > 50:
        analysis.append(f"🔴 CRITICAL: Your system has detected an unusually high attack rate of {attack_rate}%. Immediate action recommended.")
    elif attack_rate > 20:
        analysis.append(f"⚠️ WARNING: {attack_rate}% of network traffic shows suspicious patterns. Monitor closely and strengthen security measures.")
    elif attack_rate > 5:
        analysis.append(f"🟡 CAUTIOUS: {attack_rate}% attack rate detected. Continue monitoring and maintain security posture.")
    else:
        analysis.append(f"✅ System is relatively secure with only {attack_rate}% suspicious activity detected.")
    
    analysis.append(f"\nTotal predictions analyzed: {total} | Confirmed attacks: {attacks} | Normal traffic: {metrics['normal_count']}")
    
    # ========== CONFIDENCE ASSESSMENT ==========
    analysis.append("\n\nDETECTION CONFIDENCE")
    if avg_confidence >= 80:
        analysis.append(f"Average prediction confidence: {avg_confidence}% - VERY HIGH confidence in detections")
    elif avg_confidence >= 70:
        analysis.append(f"Average prediction confidence: {avg_confidence}% - HIGH confidence in detections")
    elif avg_confidence >= 50:
        analysis.append(f"Average prediction confidence: {avg_confidence}% - MODERATE confidence in detections")
    else:
        analysis.append(f"Average prediction confidence: {avg_confidence}% - LOW confidence; consider manual review")
    
    conf_breakdown = f"High confidence attacks (≥70%): {high_conf} | Medium confidence (50-70%): {medium_conf}"
    analysis.append(f"\n{conf_breakdown}")
    
    # ========== THREAT PATTERNS ==========
    analysis.append("\n\nTHREAT PATTERNS")
    if most_common:
        top_threat = most_common[0]
        threat_name = top_threat['activity_type']
        threat_count = top_threat['count']
        threat_pct = (threat_count / attacks * 100) if attacks > 0 else 0
        analysis.append(f"Most prevalent attack: {threat_name} ({threat_count} occurrences, {threat_pct:.1f}% of attacks)")
        
        if threat_pct > 50:
            analysis.append(f"⚠️ ALERT: Over 50% of attacks are '{threat_name}'. Focus on mitigating this specific threat vector.")
        elif threat_pct > 30:
            analysis.append(f"⚠️ ALERT: '{threat_name}' accounts for {threat_pct:.1f}% of attacks. Consider targeted defense strategies.")
    
    # ========== TEMPORAL ANALYSIS ==========
    hourly = metrics.get('hourly_attacks', [])
    if hourly:
        peak_hour = max(hourly, key=lambda x: x['count'])
        analysis.append(f"\n\nTEMPORAL ANALYSIS")
        analysis.append(f"Peak attack hour: {peak_hour['hour']}:00 (UTC) with {peak_hour['count']} attacks")
        analysis.append(f"💡 Consider scheduling additional monitoring or automated responses during this time window.")
    
    # ========== RECOMMENDATIONS ==========
    analysis.append("\n\nAI RECOMMENDATIONS")
    
    recommendations = []
    
    # Attack rate-based recommendations
    if attack_rate > 50:
        recommendations.append("🔴 CRITICAL: Over 50% of traffic is classified as attacks. Implement emergency security measures immediately:")
        recommendations.append("   - Enable strict firewall rules and rate limiting")
        recommendations.append("   - Activate DDoS protection mechanisms")
        recommendations.append("   - Consider restricting access to critical systems")
    elif attack_rate > 30:
        recommendations.append("🟠 URGENT: Attack rate exceeds 30%. Strengthen your defenses:")
        recommendations.append("   - Increase firewall rule strictness")
        recommendations.append("   - Deploy additional rate limiting mechanisms")
        recommendations.append("   - Review and update security policies")
    elif attack_rate > 10:
        recommendations.append("🟡 CAUTION: Notable attack rate detected (>10%). Maintain enhanced monitoring:")
        recommendations.append("   - Monitor traffic patterns closely")
        recommendations.append("   - Review security logs regularly")
    else:
        recommendations.append("✅ Attack rate is low. Maintain current security posture and continue monitoring.")
    
    # Confidence-based recommendations
    recommendations.append("")
    if avg_confidence >= 80:
        recommendations.append("🎯 HIGH CONFIDENCE DETECTIONS: Trust detection results with high confidence. Prioritize response to alerts.")
        if high_conf > 0:
            recommendations.append(f"   - {high_conf} high-confidence attacks detected - immediate investigation recommended")
    elif avg_confidence >= 70:
        recommendations.append("📊 GOOD CONFIDENCE: Detections are reliable. Act on alerts with proper verification.")
    elif avg_confidence >= 50:
        recommendations.append("⚠️ MODERATE CONFIDENCE: Review alerts before taking action. Consider manual verification of suspicious patterns.")
        recommendations.append("   - Conduct manual reviews of detected traffic")
        recommendations.append("   - Refine detection model with labeled data")
    else:
        recommendations.append("🔍 LOW CONFIDENCE: Manual review required before action. Confidence model may need retraining.")
        recommendations.append("   - Don't rely solely on automated alerts")
        recommendations.append("   - Conduct thorough manual security analysis")
        recommendations.append("   - Collect more labeled data to improve model accuracy")
    
    # Confidence breakdown recommendations
    recommendations.append("")
    if high_conf > 0:
        recommendations.append(f"📌 {high_conf} high-confidence alerts detected (≥70%): These require immediate attention and action.")
    if medium_conf > high_conf:
        recommendations.append(f"⚡ {medium_conf} medium-confidence alerts (50-70%): Review these with moderate priority.")
    if low_conf > 0:
        recommendations.append(f"📋 {low_conf} low-confidence alerts (<50%): Log and analyze for pattern detection.")
    
    # Attack pattern recommendations
    recommendations.append("")
    if most_common and len(most_common) > 0:
        top_threat = most_common[0]
        threat_name = top_threat['activity_type']
        threat_count = top_threat['count']
        threat_pct = (threat_count / attacks * 100) if attacks > 0 else 0
        
        if threat_pct > 50:
            recommendations.append(f"🎯 FOCUS AREA: '{threat_name}' represents {threat_pct:.1f}% of attacks.")
            recommendations.append(f"   - Implement specialized defenses against {threat_name}")
            recommendations.append(f"   - Deploy targeted monitoring for this attack vector")
        elif threat_pct > 30:
            recommendations.append(f"📌 SIGNIFICANT THREAT: '{threat_name}' accounts for {threat_pct:.1f}% of attacks.")
            recommendations.append(f"   - Prioritize mitigation strategies for {threat_name}")
            recommendations.append(f"   - Consider specialized security tools for this threat type")
        else:
            recommendations.append(f"✓ Multiple attack types detected. '{threat_name}' is the most common at {threat_pct:.1f}%.")
    
    # Combined attack + confidence recommendations
    recommendations.append("")
    if attack_rate > 20 and avg_confidence < 60:
        recommendations.append("⚠️ COMBINED ALERT: High attack rate with low confidence - model may be overwhelmed or miscalibrated.")
        recommendations.append("   - Review and recalibrate detection thresholds")
        recommendations.append("   - Consider retraining with recent attack samples")
    elif attack_rate < 10 and avg_confidence >= 80:
        recommendations.append("✅ OPTIMAL STATUS: Low attack rate with high detection confidence. System is functioning well.")
    elif attack_rate > 20 and avg_confidence >= 80:
        recommendations.append("🎯 EFFECTIVE DETECTION: Despite high attacks, detection is highly confident. Alerts are reliable.")
    
    # Default message if no specific condition matched
    if len(recommendations) <= 5:
        recommendations.append("\n✓ Continue monitoring. System is operating normally.")
    
    for rec in recommendations:
        analysis.append(f"\n{rec}")
    
    return "\n".join(analysis)
