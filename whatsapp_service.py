# whatsapp_service.py - ENHANCED with DRY_RUN mode and OTP support
import requests
import json
import logging
from config import Config

logger = logging.getLogger(__name__)

class WhatsAppService:
    def __init__(self):
        self.token = Config.WHATSAPP_TOKEN
        self.phone_id = Config.WHATSAPP_PHONE_ID
        self.dry_run = Config.WHATSAPP_DRY_RUN
        self.base_url = f"https://graph.facebook.com/v17.0/{self.phone_id}/messages"

    def send_message(self, to_phone, message):
        """Send text message (respects DRY_RUN mode)"""
        # DRY_RUN mode: just log and return success
        if self.dry_run:
            logger.info("=" * 60)
            logger.info("📱 WHATSAPP DRY_RUN MODE (no actual send)")
            logger.info(f"To: {to_phone}")
            logger.info(f"Message: {message}")
            logger.info("=" * 60)
            print(f"[DRY_RUN] WhatsApp to {to_phone}: {message}")
            return True
        
        # Real send
        if not self.token or not self.phone_id:
            logger.error("WhatsApp credentials not configured")
            return False
            
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "text",
            "text": {"body": message}
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            logger.info(f"✅ WhatsApp sent successfully to {to_phone}")
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error sending WhatsApp message: {e}")
            return False

    def send_otp(self, to_phone, otp):
        """Send OTP for password reset"""
        message = (
            f"🔐 Your OTP for password reset is: {otp}\n\n"
            f"This code will expire in {Config.OTP_EXP_MINUTES} minutes.\n\n"
            f"If you didn't request this, please ignore."
        )
        return self.send_message(to_phone, message)

    def send_template_message(self, to_phone, template_name, parameters):
        """Send template message"""
        if self.dry_run:
            logger.info(f"[DRY_RUN] Template '{template_name}' to {to_phone}: {parameters}")
            return True
            
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        data = {
            "messaging_product": "whatsapp",
            "to": to_phone,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": "en"},
                "components": [
                    {
                        "type": "body",
                        "parameters": parameters
                    }
                ]
            }
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending WhatsApp template message: {e}")
            return False

    def send_absence_alert(self, parent_phone, student_name, days_absent):
        """Send absence notification"""
        message = f"⚠️ Alert: {student_name} has been absent for {days_absent} consecutive days. Please contact the school if needed."
        return self.send_message(parent_phone, message)

    def send_achievement_alert(self, parent_phone, student_name, achievement):
        """Send achievement notification"""
        message = f"🎉 Congratulations! {student_name} has achieved: {achievement}. Well done!"
        return self.send_message(parent_phone, message)