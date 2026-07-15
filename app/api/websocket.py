"""
WebSocket endpoint handler for FastAPI.

This module provides the WebSocket endpoint for real-time voice communication
supporting multiple concurrent user connections with capacity management.

Features:
- Optional noise suppression (configurable)
- ai-coustics AIC speech enhancement (noise reduction + clarity)
- SmartTurn v3 ML-based end-of-turn detection
- Emotion detection via MSP-PODCAST + Gemini
"""

import uuid
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


async def websocket_endpoint(websocket: WebSocket) -> None:
    """FastAPI WebSocket endpoint for Voice Assistant with concurrent connection support.

    This endpoint handles multiple WebSocket connections simultaneously with:
    - Connection capacity limits (20 max for Lightsail)
    - Session tracking and management
    - Heartbeat monitoring for stale connections
    - Isolated VoiceAssistant instance per connection
    - SmartTurn v3 ML-based end-of-turn detection
    - Optional noise suppression (configurable)
    - ai-coustics AIC speech enhancement (optional)

    Args:
        websocket: FastAPI WebSocket connection
    """
    session_id = str(uuid.uuid4())[:8]
    voice_assistant = None  # bound in the try below; checked in finally for insight capture
    logger.info(f"[Session {session_id}] New WebSocket connection attempt")

    # Read persona_id from WebSocket query params (appended by /connect endpoint)
    persona_id = websocket.query_params.get("persona_id", "")
    if persona_id:
        logger.info(f"[Session {session_id}] Persona requested: {persona_id}")

    # Import connection manager
    from app.core.connection_manager import connection_manager

    # Try to accept connection (may reject if at capacity)
    await connection_manager.connect(websocket, session_id)

    # If we reach here, connection was accepted
    try:
        # Import here to avoid circular imports
        from app.core.server import voice_assistant_server
        from app.core.voice_assistant import VoiceAssistant
        from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport
        from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
        from pipecat.serializers.protobuf import ProtobufFrameSerializer
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        # MinWordsInterruptionStrategy moved to voice_assistant.py (PipelineParams level)

        # Get configuration from config.yaml
        # Note: server_config from voice_assistant_server may not have all keys if initialized
        # before config was loaded, so read directly from full config
        full_config = voice_assistant_server.config or {}
        server_config = full_config.get("server", {})
        vad_config = server_config.get("vad", {})
        interruption_config = server_config.get("interruption", {})

        # Log raw config to debug why config values aren't being applied
        logger.info(f"[Session {session_id}] 📋 Raw server_config keys: {list(server_config.keys())}")
        logger.info(f"[Session {session_id}] 📋 Raw vad_config: {vad_config}")

        # AI noise cancellation (Koala/Krisp/AIC) removed — raw audio goes straight
        # to STT. Deepgram Flux is noise-robust and does its own turn detection, so
        # the filter stage added latency + license complexity for no gain.

        # Stricter VAD settings to prevent false barge-ins from background noise
        # MinWordsInterruptionStrategy (below) provides additional filtering
        vad_params = VADParams(
            confidence=vad_config.get("confidence", 0.7),     # HIGHER - only trigger on clear speech
            start_secs=vad_config.get("start_secs", 0.5),      # SLOWER - require 500ms of speech (filters noise)
            stop_secs=vad_config.get("stop_secs", 1.0),        # Wait 1s of silence before ending utterance
            min_volume=vad_config.get("min_volume", 0.65),     # HIGHER - ignore quiet background noise
        )
        vad_analyzer = SileroVADAnalyzer(params=vad_params)

        # Interruption strategy is configured in voice_assistant.py via PipelineParams
        # (MinWordsInterruptionStrategy is a pipeline-level param, not transport-level)

        # Register VAD analyzer for runtime parameter changes (noise cancellation toggle)
        connection_manager.register_vad_analyzer(session_id, vad_analyzer)

        logger.info(
            f"[Session {session_id}] 🎤 VAD configured: confidence={vad_params.confidence}, "
            f"start_secs={vad_params.start_secs}, stop_secs={vad_params.stop_secs}, "
            f"min_volume={vad_params.min_volume}"
        )

        # ===== TURN DETECTION =====
        # Deepgram Flux does native end-of-turn detection (StartOfTurn/EndOfTurn
        # events with confidence thresholds), so the SmartTurn v3 ONNX analyzer is
        # redundant when Flux is the STT provider — skip it regardless of config.
        smart_turn_config = server_config.get("smart_turn", {})
        stt_provider = full_config.get("stt", {}).get("provider", "")
        turn_analyzer = None
        if stt_provider == "deepgram_flux":
            logger.info(f"[Session {session_id}] 🧠 Turn detection: Deepgram Flux native EOT (SmartTurn skipped)")
        elif smart_turn_config.get("enabled", False):
            try:
                from app.processors.logging_turn_analyzer import LoggingSmartTurnAnalyzer
                cpu_count = smart_turn_config.get("cpu_count", 1)
                turn_analyzer = LoggingSmartTurnAnalyzer(
                    cpu_count=cpu_count,
                    session_id=session_id
                )
                logger.info(f"[Session {session_id}] 🧠 SmartTurn v3: ENABLED at transport level (ONNX ML model)")
            except Exception as e:
                logger.error(f"[Session {session_id}] 🧠 SmartTurn v3: Failed to initialize: {e}")
                logger.info(f"[Session {session_id}] 🧠 Falling back to transcription-based detection")
        else:
            logger.info(f"[Session {session_id}] 🧠 SmartTurn v3: DISABLED (using transcription-based detection)")

        # Create transport parameters for this connection.
        # No audio_in_filter: noise cancellation removed (raw 16 kHz audio to STT).
        transport_params = FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            vad_analyzer=vad_analyzer,
            vad_audio_passthrough=True,
            serializer=ProtobufFrameSerializer(),
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,  # Chatterbox TTS outputs 24 kHz
            turn_analyzer=turn_analyzer,  # None when Flux handles end-of-turn natively
        )

        filter_desc = "None (raw audio — noise cancellation removed)"
        logger.info(f"[Session {session_id}] 🔧 Transport configured (no audio filter)")

        # Create transport for this specific connection
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=transport_params,
        )

        # Look up persona config if persona_id was provided
        persona_config = None
        if persona_id:
            personas = full_config.get("personas", {}).get("agents", {})
            if persona_id in personas:
                persona_config = personas[persona_id]
                logger.info(f"[Session {session_id}] Loaded persona: {persona_config.get('name', persona_id)} (voice_id={'SET' if persona_config.get('voice_id') else 'DEFAULT'})")
            else:
                logger.warning(f"[Session {session_id}] Persona '{persona_id}' not found, using default")

        # Create dedicated VoiceAssistant instance for this session
        voice_assistant = VoiceAssistant(voice_assistant_server.config, persona_config=persona_config)
        logger.info(f"[Session {session_id}] VoiceAssistant instance created")

        # Log emotion detection state for this session
        emotion_enabled = server_config.get("emotion_detection_enabled", True)
        logger.info(
            f"[Session {session_id}] [EMOTION-DIAG] Session emotion config: "
            f"enabled={emotion_enabled}, "
            f"tone_processor_enabled={voice_assistant.tone_processor.enabled}, "
            f"hybrid_mode={voice_assistant.tone_processor.use_hybrid_mode}"
        )

        # Log complete audio processing pipeline
        smart_turn_desc = "SmartTurn v3 (transport)" if turn_analyzer else "Transcription-based"
        logger.info(
            f"[Session {session_id}] 📊 AUDIO PIPELINE SUMMARY:\n"
            f"  ┌─ Input: Microphone (16kHz)\n"
            f"  ├─ Filters: {filter_desc}\n"
            f"  ├─ VAD: Silero (conf={vad_params.confidence}, start={vad_params.start_secs}s, vol={vad_params.min_volume})\n"
            f"  ├─ Turn Detection: {smart_turn_desc}\n"
            f"  ├─ STT Mute: ALWAYS (blocks VAD/STT during bot speech)\n"
            f"  ├─ STT: Deepgram Nova-3\n"
            f"  ├─ LLM: Groq Llama-3.3-70b\n"
            f"  └─ TTS: ElevenLabs (24kHz)"
        )

        # Run the voice assistant pipeline for this connection
        # This will block until the connection closes
        await voice_assistant.run(transport, handle_sigint=False)

        logger.info(f"[Session {session_id}] Session completed normally")

    except WebSocketDisconnect:
        logger.info(f"[Session {session_id}] Client disconnected")
    except Exception as e:
        logger.error(f"[Session {session_id}] Exception in WebSocket endpoint: {e}")
    finally:
        # Clean up connection in manager
        connection_manager.disconnect(session_id)
        logger.info(
            f"[Session {session_id}] Connection closed. "
            f"Active sessions: {connection_manager.get_active_session_count()}"
        )

        # Post-call insight capture (Ather's fourth pillar). Runs AFTER the session is
        # closed, so it adds zero conversational latency; capture_insight itself is
        # fail-safe (never raises), and this guard keeps session teardown bulletproof.
        try:
            cm = voice_assistant.conversation_manager if voice_assistant else None
            if cm and cm.context and cm.context.messages:
                transcript = "\n".join(
                    f"{'CALLER' if m.get('role') == 'user' else 'AGENT'}: {m.get('content', '')}"
                    for m in cm.context.messages
                    if m.get("role") in ("user", "assistant") and str(m.get("content", "")).strip()
                )
                if transcript.strip():
                    import os as _os
                    from app.services.insight_capture import capture_insight
                    await capture_insight(
                        transcript_text=transcript,
                        call_id=session_id,
                        groq_api_key=_os.getenv("GROQ_API_KEY", ""),
                        persona_id=cm.current_persona_id,
                    )
        except Exception as insight_err:  # noqa: BLE001 - never break teardown
            logger.warning(f"[Session {session_id}] insight capture skipped: {insight_err}")
