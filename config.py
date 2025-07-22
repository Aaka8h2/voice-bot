import os
import json
import asyncio
import hashlib
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
import aiofiles
from dotenv import load_dotenv

load_dotenv()

# ========== CONFIGURATION CLASSES ==========
@dataclass
class BotSettings:
    BOT_TOKEN: str = os.getenv('BOT_TOKEN', '')
    ADMIN_IDS: List[int] = None
    BOT_USERNAME: str = os.getenv('BOT_USERNAME', '@your_bot')
    UPI_ID: str = os.getenv('UPI_ID', 'your-upi@paytm')
    UPI_NAME: str = os.getenv('UPI_NAME', 'Your Name')
    ENCRYPTION_KEY: str = os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode())
    JWT_SECRET: str = os.getenv('JWT_SECRET', 'ultra-secret-2025')
    
    def __post_init__(self):
        if self.ADMIN_IDS is None:
            admin_str = os.getenv('ADMIN_IDS', '123456789')
            self.ADMIN_IDS = [int(x.strip()) for x in admin_str.split(',') if x.strip()]

@dataclass
class ChannelConfig:
    USERNAME: str = os.getenv('CHANNEL_USERNAME', '@your_channel')
    ID: int = int(os.getenv('CHANNEL_ID', '-1001234567890'))
    NAME: str = os.getenv('CHANNEL_NAME', 'Main Channel')

@dataclass
class FeatureFlags:
    VOICE_CLONING: bool = os.getenv('ENABLE_VOICE_CLONING', 'true').lower() == 'true'
    AI_ANALYSIS: bool = os.getenv('ENABLE_AI_ANALYSIS', 'true').lower() == 'true'
    BIOMETRIC_AUTH: bool = os.getenv('ENABLE_BIOMETRIC_AUTH', 'true').lower() == 'true'
    FREE_DAILY_LIMIT: int = int(os.getenv('MAX_DAILY_CONVERSIONS_FREE', '50'))
    PREMIUM_DAILY_LIMIT: int = int(os.getenv('MAX_DAILY_CONVERSIONS_PREMIUM', '1000'))

# Global configuration instances
settings = BotSettings()
channel_config = ChannelConfig()
features = FeatureFlags()

# ========== PREMIUM PLANS ==========
PREMIUM_PLANS = {
    'monthly': {
        'name': '💎 Premium Monthly',
        'price': 99,
        'duration': 30,
        'features': {
            'daily_limit': 1000,
            'voice_cloning': True,
            'ai_effects': True,
            'priority_support': True,
            'analytics': True
        }
    },
    'yearly': {
        'name': '👑 Premium Yearly',
        'price': 999,
        'duration': 365,
        'features': {
            'daily_limit': 2000,
            'voice_cloning': True,
            'ai_effects': True,
            'priority_support': True,
            'analytics': True,
            'custom_training': True,
            'api_access': True
        }
    }
}

# ========== SUPPORTED LANGUAGES ==========
LANGUAGES = {
    'en': '🇺🇸 English', 'hi': '🇮🇳 Hindi', 'es': '🇪🇸 Spanish', 'fr': '🇫🇷 French',
    'de': '🇩🇪 German', 'it': '🇮🇹 Italian', 'pt': '🇵🇹 Portuguese', 'ru': '🇷🇺 Russian',
    'ja': '🇯🇵 Japanese', 'ko': '🇰🇷 Korean', 'zh': '🇨🇳 Chinese', 'ar': '🇸🇦 Arabic',
    'th': '🇹🇭 Thai', 'vi': '🇻🇳 Vietnamese'
}

# ========== ADVANCED DATABASE MANAGER ==========
class UltraDatabase:
    def __init__(self):
        self.data_dir = 'data'
        self.encryption_key = Fernet(settings.ENCRYPTION_KEY.encode())
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Database files
        self.files = {
            'users': f'{self.data_dir}/users.json',
            'premium': f'{self.data_dir}/premium.json',
            'payments': f'{self.data_dir}/payments.json',
            'voice_models': f'{self.data_dir}/voice_models.json',
            'analytics': f'{self.data_dir}/analytics.json',
            'settings': f'{self.data_dir}/bot_settings.json',
            'banned': f'{self.data_dir}/banned_users.json'
        }
        
        self.init_database()
    
    def init_database(self):
        """Initialize all database files with default data"""
        default_data = {
            'users': {},
            'premium': {},
            'payments': {},
            'voice_models': {},
            'analytics': {
                'total_users': 0,
                'total_conversions': 0,
                'daily_stats': {},
                'language_usage': {},
                'last_updated': datetime.now().isoformat()
            },
            'settings': {
                'bot_enabled': True,
                'maintenance_mode': False,
                'created_at': datetime.now().isoformat(),
                'version': '2.0.0'
            },
            'banned': {}
        }
        
        for key, filepath in self.files.items():
            if not os.path.exists(filepath):
                with open(filepath, 'w') as f:
                    json.dump(default_data[key], f, indent=2, default=str)
    
    async def read_data(self, table: str) -> Dict:
        """Read data from JSON file"""
        try:
            if table not in self.files:
                return {}
            async with aiofiles.open(self.files[table], 'r') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            print(f"Error reading {table}: {e}")
            return {}
    
    async def write_data(self, table: str, data: Dict):
        """Write data to JSON file"""
        try:
            if table not in self.files:
                return
            async with aiofiles.open(self.files[table], 'w') as f:
                await f.write(json.dumps(data, indent=2, default=str))
        except Exception as e:
            print(f"Error writing {table}: {e}")
    
    # ========== USER MANAGEMENT ==========
    async def create_user(self, user_id: int, user_data: Dict) -> Dict:
        """Create comprehensive user profile"""
        users = await self.read_data('users')
        
        user_profile = {
            'user_id': user_id,
            'basic_info': {
                'first_name': user_data.get('first_name', ''),
                'username': user_data.get('username', ''),
                'language_code': user_data.get('language_code', 'en'),
                'phone': user_data.get('phone'),
                'joined_at': datetime.now().isoformat()
            },
            'preferences': {
                'tts_language': 'en',
                'voice_model': 'neural',
                'voice_speed': 1.0,
                'voice_effect': 'none',
                'notifications': True
            },
            'subscription': {
                'plan': 'free',
                'expires_at': None,
                'features': ['basic_tts']
            },
            'usage_stats': {
                'daily_usage': 0,
                'total_usage': 0,
                'last_reset': datetime.now().isoformat(),
                'last_active': datetime.now().isoformat()
            },
            'security': {
                'two_factor': False,
                'biometric_hash': None,
                'last_login': datetime.now().isoformat()
            },
            'ai_data': {
                'voice_samples': [],
                'custom_models': [],
                'usage_patterns': [],
                'sentiment_history': []
            }
        }
        
        users[str(user_id)] = user_profile
        await self.write_data('users', users)
        
        # Update analytics
        await self.increment_total_users()
        
        return user_profile
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user profile"""
        users = await self.read_data('users')
        user = users.get(str(user_id))
        if user:
            # Update last active
            user['usage_stats']['last_active'] = datetime.now().isoformat()
            users[str(user_id)] = user
            await self.write_data('users', users)
        return user
    
    async def update_user(self, user_id: int, updates: Dict):
        """Update user data"""
        users = await self.read_data('users')
        if str(user_id) in users:
            # Deep merge updates
            user = users[str(user_id)]
            for key, value in updates.items():
                if isinstance(value, dict) and key in user:
                    user[key].update(value)
                else:
                    user[key] = value
            
            user['usage_stats']['last_active'] = datetime.now().isoformat()
            users[str(user_id)] = user
            await self.write_data('users', users)
    
    async def get_all_users(self) -> Dict:
        """Get all users"""
        return await self.read_data('users')
    
    # ========== PREMIUM MANAGEMENT ==========
    async def add_premium(self, user_id: int, plan: str):
        """Add premium subscription"""
        premium_data = await self.read_data('premium')
        plan_info = PREMIUM_PLANS[plan]
        
        expiry_date = datetime.now() + timedelta(days=plan_info['duration'])
        
        premium_data[str(user_id)] = {
            'user_id': user_id,
            'plan': plan,
            'activated_at': datetime.now().isoformat(),
            'expires_at': expiry_date.isoformat(),
            'features': plan_info['features'],
            'status': 'active'
        }
        
        await self.write_data('premium', premium_data)
        
        # Update user subscription
        await self.update_user(user_id, {
            'subscription': {
                'plan': plan,
                'expires_at': expiry_date.isoformat(),
                'features': list(plan_info['features'].keys())
            }
        })
    
    async def is_premium(self, user_id: int) -> bool:
        """Check if user has active premium"""
        premium_data = await self.read_data('premium')
        user_premium = premium_data.get(str(user_id))
        
        if not user_premium:
            return False
        
        expiry_date = datetime.fromisoformat(user_premium['expires_at'])
        if datetime.now() > expiry_date:
            # Remove expired premium
            del premium_data[str(user_id)]
            await self.write_data('premium', premium_data)
            await self.update_user(user_id, {
                'subscription': {'plan': 'free', 'expires_at': None, 'features': ['basic_tts']}
            })
            return False
        
        return True
    
    # ========== PAYMENT MANAGEMENT ==========
    async def create_payment(self, user_id: int, plan: str, amount: int) -> str:
        """Create payment request"""
        payments = await self.read_data('payments')
        payment_id = f"pay_{user_id}_{int(datetime.now().timestamp())}"
        
        payments[payment_id] = {
            'payment_id': payment_id,
            'user_id': user_id,
            'plan': plan,
            'amount': amount,
            'status': 'pending',
            'utr': None,
            'screenshot_file_id': None,
            'created_at': datetime.now().isoformat(),
            'reviewed_by': None,
            'reviewed_at': None
        }
        
        await self.write_data('payments', payments)
        return payment_id
    
    async def update_payment(self, payment_id: str, updates: Dict):
        """Update payment data"""
        payments = await self.read_data('payments')
        if payment_id in payments:
            payments[payment_id].update(updates)
            await self.write_data('payments', payments)
    
    async def get_payment(self, payment_id: str) -> Optional[Dict]:
        """Get payment data"""
        payments = await self.read_data('payments')
        return payments.get(payment_id)
    
    async def get_pending_payments(self) -> List[Dict]:
        """Get all pending payments for admin review"""
        payments = await self.read_data('payments')
        return [p for p in payments.values() if p['status'] == 'pending' and p['utr'] and p['screenshot_file_id']]
    
    # ========== VOICE MODEL MANAGEMENT ==========
    async def save_voice_model(self, user_id: int, model_data: Dict) -> str:
        """Save custom voice model"""
        voice_models = await self.read_data('voice_models')
        model_id = f"voice_{user_id}_{int(datetime.now().timestamp())}"
        
        voice_models[model_id] = {
            'model_id': model_id,
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'quality_score': model_data.get('quality_score', 0.8),
            'sample_count': model_data.get('sample_count', 0),
            'model_size': model_data.get('model_size', 0),
            'status': 'ready',
            'usage_count': 0
        }
        
        await self.write_data('voice_models', voice_models)
        
        # Update user AI data
        await self.update_user(user_id, {
            'ai_data': {'custom_models': [model_id]}
        })
        
        return model_id
    
    async def get_user_voice_models(self, user_id: int) -> List[Dict]:
        """Get user's voice models"""
        voice_models = await self.read_data('voice_models')
        return [model for model in voice_models.values() if model['user_id'] == user_id]
    
    # ========== ANALYTICS ==========
    async def increment_total_users(self):
        """Increment total users count"""
        analytics = await self.read_data('analytics')
        analytics['total_users'] = analytics.get('total_users', 0) + 1
        await self.write_data('analytics', analytics)
    
    async def log_conversion(self, user_id: int, language: str, model_used: str):
        """Log TTS conversion"""
        analytics = await self.read_data('analytics')
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Update daily stats
        if 'daily_stats' not in analytics:
            analytics['daily_stats'] = {}
        if today not in analytics['daily_stats']:
            analytics['daily_stats'][today] = {'conversions': 0, 'unique_users': set()}
        
        analytics['daily_stats'][today]['conversions'] += 1
        analytics['daily_stats'][today]['unique_users'].add(user_id)
        
        # Update language usage
        if 'language_usage' not in analytics:
            analytics['language_usage'] = {}
        analytics['language_usage'][language] = analytics['language_usage'].get(language, 0) + 1
        
        # Convert sets to lists for JSON serialization
        for date_stats in analytics['daily_stats'].values():
            if isinstance(date_stats['unique_users'], set):
                date_stats['unique_users'] = list(date_stats['unique_users'])
        
        analytics['total_conversions'] = analytics.get('total_conversions', 0) + 1
        analytics['last_updated'] = datetime.now().isoformat()
        
        await self.write_data('analytics', analytics)
        
        # Update user usage
        await self.increment_user_usage(user_id)
    
    async def increment_user_usage(self, user_id: int):
        """Increment user's usage counters"""
        user = await self.get_user(user_id)
        if user:
            # Check if daily usage needs reset
            last_reset = datetime.fromisoformat(user['usage_stats']['last_reset'])
            if last_reset.date() < datetime.now().date():
                daily_usage = 1
                last_reset = datetime.now().isoformat()
            else:
                daily_usage = user['usage_stats']['daily_usage'] + 1
            
            await self.update_user(user_id, {
                'usage_stats': {
                    'daily_usage': daily_usage,
                    'total_usage': user['usage_stats']['total_usage'] + 1,
                    'last_reset': last_reset
                }
            })
    
    async def get_user_daily_usage(self, user_id: int) -> int:
        """Get user's current daily usage"""
        user = await self.get_user(user_id)
        if not user:
            return 0
        
        last_reset = datetime.fromisoformat(user['usage_stats']['last_reset'])
        if last_reset.date() < datetime.now().date():
            return 0
        
        return user['usage_stats']['daily_usage']
    
    # ========== BAN MANAGEMENT ==========
    async def ban_user(self, user_id: int, reason: str = "", admin_id: int = None):
        """Ban user"""
        banned = await self.read_data('banned')
        banned[str(user_id)] = {
            'user_id': user_id,
            'banned_at': datetime.now().isoformat(),
            'reason': reason,
            'banned_by': admin_id
        }
        await self.write_data('banned', banned)
    
    async def unban_user(self, user_id: int):
        """Unban user"""
        banned = await self.read_data('banned')
        if str(user_id) in banned:
            del banned[str(user_id)]
            await self.write_data('banned', banned)
    
    async def is_banned(self, user_id: int) -> bool:
        """Check if user is banned"""
        banned = await self.read_data('banned')
        return str(user_id) in banned
    
    # ========== DATA EXPORT ==========
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        for table_name in self.files.keys():
            try:
                data = await self.read_data(table_name)
                stats[table_name] = {
                    'records': len(data),
                    'file_size': os.path.getsize(self.files[table_name]) if os.path.exists(self.files[table_name]) else 0
                }
            except:
                stats[table_name] = {'records': 0, 'file_size': 0}
        
        return stats

# Global database instance
db = UltraDatabase()

# ========== UTILITY FUNCTIONS ==========
def encrypt_sensitive_data(data: str) -> str:
    """Encrypt sensitive user data"""
    try:
        f = Fernet(settings.ENCRYPTION_KEY.encode())
        return f.encrypt(data.encode()).decode()
    except:
        return data

def decrypt_sensitive_data(encrypted_data: str) -> str:
    """Decrypt sensitive user data"""
    try:
        f = Fernet(settings.ENCRYPTION_KEY.encode())
        return f.decrypt(encrypted_data.encode()).decode()
    except:
        return encrypted_data

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def format_duration(seconds: int) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{hours}h {remaining_minutes}m"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis"""
    return text if len(text) <= max_length else text[:max_length-3] + "..."
