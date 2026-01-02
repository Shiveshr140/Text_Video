import os
from celery import Celery

# Manim pipelines (existing)
from simple_app import text_to_video as manim_text_to_video
from simple_app import code_to_video as manim_code_to_video
from simple_app import query_to_animated_video_v3

# Remotion pipelines (new)
try:
    from test_code_video import code_to_video as remotion_code_to_video
    from test_text_video import text_to_video as remotion_text_to_video
    REMOTION_AVAILABLE = True
except ImportError:
    print("⚠️  Remotion pipelines not available (test_code_video.py or test_text_video.py not found)")
    REMOTION_AVAILABLE = False

# Redis configuration from your org
REDIS_HOST = 'redis-10936.c264.ap-south-1-1.ec2.redns.redis-cloud.com'
REDIS_PORT = '10936'
REDIS_PASSWORD = 'HXcegHKO8Wg92K27iT6azObk1ayuULPP'
REDIS_DB = '0'  # Using DB 0 as in the Go example (default)

# Build Redis URL with key prefix for namespacing
REDIS_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'

# Configure Celery (simplified - no global_keyprefix to avoid backend issues)
celery_app = Celery(
    'manim-tasks',  # App name acts as prefix
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Configure task settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@celery_app.task(bind=True)
def task_text_to_video(self, text, language="english", renderer="manim"):
    """
    Text to video - supports both Manim and Remotion
    
    Args:
        text: Text content to convert
        language: Audio language (default: english)
        renderer: "manim" or "remotion" (default: manim)
    """
    output_name = f"text_job_{self.request.id}"
    
    if renderer == "remotion" and REMOTION_AVAILABLE:
        result = remotion_text_to_video(text, output_name=output_name)
    else:
        # Default to Manim
        result = manim_text_to_video(text, output_name=output_name, audio_language=language)
    
    return result

@celery_app.task(bind=True)
def task_code_to_video(self, code, language="english", code_language="python", renderer="manim"):
    """
    Code to video - supports both Manim and Remotion
    
    Args:
        code: Code content to explain
        language: Audio language (default: english)
        code_language: Programming language of the code (default: python)
        renderer: "manim" or "remotion" (default: manim)
    """
    output_name = f"code_job_{self.request.id}"
    
    if renderer == "remotion" and REMOTION_AVAILABLE:
        result = remotion_code_to_video(code, language=code_language, output_name=output_name)
    else:
        # Default to Manim
        result = manim_code_to_video(code, output_name=output_name, audio_language=language)
    
    return result

@celery_app.task(bind=True)
def task_query_to_video(self, query, language="english"):
    """
    Query to video - ALWAYS uses Manim (animated explanations)
    
    Args:
        query: Natural language query
        language: Audio language (default: english)
    """
    output_name = f"query_job_{self.request.id}"
    result = query_to_animated_video_v3(query, output_name=output_name, audio_language=language)
    return result
