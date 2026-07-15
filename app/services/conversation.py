"""Conversation Manager for the Voice Assistant.

This module orchestrates the conversation flow and manages LLM interactions.
"""

import os
import re
from typing import Any, Callable, Dict, List, Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import TTSSpeakFrame, EndFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.llm_service import FunctionCallParams, LLMService

# Universal context system (pipecat 0.0.98)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    LLMAssistantAggregatorParams,
)

from app.services.input_analyzer import InputAnalyzer
from app.services.groq_llm_service import GroqLLMService
from app.services.tally_submission import TallySubmissionService
from app.services.rag import LightRAGService
from app.utils.validation import validate_email, spell_out_email

# LightRAG's built-in "no relevant context" phrasings. When retrieval finds nothing
# relevant, LightRAG's generator emits one of these instead of an answer. We detect it
# and convert it into an explicit compliance refusal, rather than handing a hedge back
# to the agent as if it were grounded label content. Kept tight to avoid matching real
# answers (a grounded reply about the label won't contain these exact phrases).
_RAG_NO_CONTEXT_RE = re.compile(
    r"(do(?:es)?\s*n['’]?t?\s+have enough information|"
    r"do(?:es)? not have enough information|"
    r"not able to provide an answer|"
    r"unable to provide an answer|"
    r"\[no-?context\]|"
    r"no (?:relevant )?context (?:is )?(?:available|provided|found))",
    re.IGNORECASE,
)

# High-stakes factual-lookup routing lives in ONE place shared with the eval
# harness, so the two can never diverge (see app/services/rag_routing.py).
from app.services.rag_routing import retrieval_policy

# NOTE: SmartTurn v3 was removed here — Deepgram Flux does native end-of-turn
# detection, so this eager import (which pulled ~600MB of PyTorch at startup) is
# gone. The nova-3 fallback path in websocket.py still imports the analyzer
# lazily, only when that path is actually used.

# Voice constant (default only - actual voice is controlled by ToneAwareProcessor)
DEFAULT_VOICE = "aura-2-athena-en"  # Natural, clear female voice - default


class ConversationManager:
    """Service responsible for managing conversation flow and LLM interactions."""

    # Agent roster template injected into every persona's system prompt
    # so each agent knows about all other agents and can transfer calls
    AGENT_ROSTER_TEMPLATE = """
    AGENT TRANSFER SYSTEM:
    You are part of a team of {agent_count} AI agents. You can transfer the caller to another agent at any time by calling transfer_to_agent(agent_id). Use this when:
    - The user's question is better handled by another agent's specialty
    - The user explicitly asks to talk to a specific agent
    - You're recommending another agent and the user agrees

    YOUR TEAM:
    {agent_list}

    TRANSFER RULES:
    1. Say a brief connecting line BEFORE the function call, like "Let me connect you with [Name], they're great at this." — keep it to ONE short sentence.
    2. Then call transfer_to_agent with the agent_id.
    3. CRITICAL: After calling transfer_to_agent, say NOTHING. Do not add any text after the function call. The new agent handles everything from here.
    4. Do NOT try to answer questions outside your expertise when a better-suited agent exists — transfer instead.
    5. You can still answer general questions yourself — only transfer when the topic clearly needs a specialist.
    """

    # 20 natural, conversational thinking phrases that sound more human
    THINKING_PHRASES = [
        "Umm, let me check that.",
        "Oh, let me look that up for you.",
        "Give me a sec.",
        "Hmm, let me find that.",
        "One moment.",
        "Let me see.",
        "Ah, let me search for that.",
        "Okay, checking now.",
        "Let me pull that up.",
        "Umm, searching.",
        "Yeah, let me find that.",
        "Hold on.",
        "Let me look into that.",
        "Hmm, one sec.",
        "Okay, let me check.",
        "Searching for that now.",
        "Let me grab that info.",
        "Just a moment.",
        "Alright, looking that up.",
        "Let me find that for you.",
    ]

    def __init__(self,
                 input_analyzer: InputAnalyzer,
                 llm_config: Dict[str, Any] = None,
                 language_config: Dict[str, Any] = None,
                 smart_turn_config: Dict[str, Any] = None,
                 personas_config: Dict[str, Any] = None,
                 current_persona_id: str = "",
                 rag_config: Dict[str, Any] = None):
        """Initialize the Conversation Manager."""
        self.input_analyzer = input_analyzer
        self.llm_config = llm_config or {}
        self.language_config = language_config or {}
        self.smart_turn_config = smart_turn_config or {}
        self.llm_service = None
        self.tts_service = None
        self.context_aggregator = None
        self.context = None
        self._thinking_phrase_index = 0

        # Agent transfer state
        self.personas_config = personas_config or {}
        self.current_persona_id = current_persona_id
        self._on_agent_transfer: Optional[Callable] = None
        self._on_a2ui: Optional[Callable] = None
        self._handoff_trail: List[Dict[str, str]] = []  # routing history for the timeline card

        # RAG (LightRAG) state — audience-scoped knowledge retrieval.
        # Services are built lazily per scope ("hcp"/"patient") and cached, so a
        # live transfer (e.g. Claire -> Sophie) resolves to the correct workspace.
        self.rag_config = rag_config or {}
        self._rag_services: Dict[str, Any] = {}

        # Appointment booking state
        self._booking_in_progress = False
        self.tally_service = TallySubmissionService()

        logger.info("Initialized Conversation Manager")

    def initialize_llm(self) -> LLMService:
        """Initialize the LLM service.

        Returns:
            The initialized LLM service
        """
        api_key = self.llm_config.get("api_key")
        if not api_key:
            raise ValueError("LLM API key is required")

        provider = self.llm_config.get("provider", "groq")

        if provider == "deepseek":
            model = self.llm_config.get("model", "deepseek-chat")
            self.llm_service = GroqLLMService(
                api_key=api_key,
                model=model,
                base_url="https://api.deepseek.com",
            )
            logger.info(f"Initialized DeepSeek LLM service with model: {model}")
        elif provider == "openai":
            model = self.llm_config.get("model", "gpt-4o")
            self.llm_service = OpenAILLMService(
                api_key=api_key,
                model=model
            )
            logger.info(f"Initialized OpenAI LLM service with model: {model}")
        elif provider == "groq":
            # Groq — uses GroqLLMService which merges consecutive user
            # messages to prevent intermittent "Failed to call a function" errors
            model = self.llm_config.get("model", "llama-3.3-70b-versatile")
            self.llm_service = GroqLLMService(
                api_key=api_key,
                model=model,
            )
            logger.info(f"Initialized Groq LLM service with model: {model}")
        else:
            # Google Gemini (lazy import to avoid google.cloud.speech_v2 dependency)
            from pipecat.services.google.llm import GoogleLLMService
            model = self.llm_config.get("model", "gemini-1.5-flash-latest")
            self.llm_service = GoogleLLMService(
                api_key=api_key,
                model=model
            )
            logger.info(f"Initialized Google Gemini LLM service with model: {model}")

        # Register function handlers
        self.llm_service.register_function("end_conversation", self._handle_end_conversation)
        self.llm_service.register_function("start_appointment_booking", self._handle_start_booking)
        self.llm_service.register_function("submit_appointment", self._handle_submit_appointment)
        self.llm_service.register_function("transfer_to_agent", self._handle_transfer_to_agent)
        self.llm_service.register_function("report_adverse_event", self._handle_report_adverse_event)
        self.llm_service.register_function("call_rag_system", self._handle_call_rag_system)

        return self.llm_service

    def _get_next_thinking_phrase(self) -> str:
        """Get the next thinking phrase in the cycle.

        Returns:
            The next thinking phrase, cycling through the list linearly.
        """
        phrase = self.THINKING_PHRASES[self._thinking_phrase_index]
        self._thinking_phrase_index = (self._thinking_phrase_index + 1) % len(self.THINKING_PHRASES)
        return phrase

    def set_tts_service(self, tts_service: Any) -> None:
        """Set the TTS service for function call feedback.
        
        Args:
            tts_service: The TTS service instance
        """
        self.tts_service = tts_service

        if self.llm_service:
            # Add event handlers for function calls
            @self.llm_service.event_handler("on_function_calls_started")
            async def on_function_calls_started(service, function_calls):
                import time
                logger.info(f"🔧 FUNCTION CALL START: {function_calls} at {time.time()}")

                # Skip thinking phrase for booking/end functions
                skip_functions = ['end_conversation', 'start_appointment_booking', 'submit_appointment', 'cancel_appointment_booking', 'transfer_to_agent', 'report_adverse_event']
                if function_calls and any(func in str(call) for func in skip_functions for call in function_calls):
                    return

                if self.tts_service:
                    phrase = self._get_next_thinking_phrase()
                    await self.tts_service.queue_frame(TTSSpeakFrame(phrase))

            # Note: on_function_calls_finished not available in Pipecat 0.0.98
            # (only on_function_calls_started and on_completion_timeout are registered)

    def _strip_markdown(self, text: str) -> str:
        """Strip markdown formatting from text for voice output.
        
        Args:
            text: Text with markdown formatting
            
        Returns:
            Plain text without markdown
        """
        # Remove markdown bold/italic (**text**, *text*)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        
        # Remove markdown headers (# Header, ## Header, etc.)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove markdown links [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove markdown code blocks ```code``` and `code`
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove markdown lists (- item, * item, 1. item)
        text = re.sub(r'^[\s]*[-*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        
        return text

    async def _handle_end_conversation(self, params: FunctionCallParams) -> None:
        """Handle end conversation function call.

        When the LLM detects the user wants to end the conversation, this sends
        an EndFrame upstream to gracefully terminate the session.

        Args:
            params: Function call parameters
        """
        import asyncio
        from pipecat.frames.frames import TTSSpeakFrame

        logger.warning("🔴 End conversation function called by LLM")

        # Push farewell message directly to TTS to avoid extra LLM round
        farewell_message = "Goodbye! Thank you for visiting."
        logger.info(f"📢 Pushing farewell message to TTS: '{farewell_message}'")

        if self.tts_service:
            await self.tts_service.queue_frame(TTSSpeakFrame(farewell_message))

        # Return empty response to function to avoid LLM generating more text
        await params.result_callback("")

        # Post-call insight panel (Ather's fourth pillar): summarize the whole call
        # into themes / label gaps / competitor mentions and show it as the final card.
        # Persisted to data/insights.jsonl too, for the aggregate report.
        try:
            await self._capture_and_show_insights()
        except Exception as exc:  # noqa: BLE001 - never block session teardown
            logger.error(f"insight capture failed (non-fatal): {exc}")

        # Wait for: TTS generation + TTS playback
        # ~1s TTS generation + ~2.5s TTS playback = 3.5s total
        logger.info("⏳ Waiting 3.5 seconds for farewell TTS to complete...")
        await asyncio.sleep(3.5)
        logger.info("✅ Wait complete, sending EndFrame")

        # Push EndFrame upstream to terminate the session
        await params.llm.push_frame(EndFrame(), FrameDirection.UPSTREAM)
        logger.info("🛑 EndFrame sent - session will terminate")

    async def _capture_and_show_insights(self) -> None:
        """Summarize the call into structured insights, persist them, and push the
        insight-panel card. Best-effort; safe to fail."""
        if not (self.context and self.context.messages):
            return
        # Flatten user+assistant turns into a plain transcript for extraction.
        lines = []
        for m in self.context.messages:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in ("user", "assistant") and content and not content.startswith("["):
                lines.append(f"{role}: {content}")
        transcript = "\n".join(lines)
        if not transcript:
            return

        groq_key = os.environ.get("GROQ_API_KEY", "")
        call_id = getattr(self, "session_id", "") or "call"
        from app.services.insight_capture import capture_insight
        from app.services.a2ui.pharma_cards import insight_panel_card

        insight = await capture_insight(transcript, call_id, groq_key,
                                        persona_id=self.current_persona_id)
        if insight:
            await self._emit_a2ui(insight_panel_card(insight))

    async def _handle_start_booking(self, params: FunctionCallParams) -> None:
        """Handle start appointment booking function call.

        Initiates the appointment booking flow and sets internal state.

        Args:
            params: Function call parameters
        """
        logger.info("📅 Start appointment booking function called")

        # Set booking state flag
        self._booking_in_progress = True

        # Return a prompt to collect user information
        response = "Great! What's your first name?"
        await params.result_callback(response)

        logger.info("✅ Appointment booking flow initiated")

    async def _handle_submit_appointment(self, params: FunctionCallParams) -> None:
        """Handle appointment submission function call.

        Validates and submits the appointment data to Tally.so.

        Args:
            params: Function call parameters with first_name, last_name, email
        """
        logger.info("📋 Submit appointment function called")

        # Extract parameters
        first_name = params.arguments.get("first_name", "").strip()
        last_name = params.arguments.get("last_name", "").strip()
        email = params.arguments.get("email", "").strip()

        logger.info(f"Appointment details: {first_name} {last_name} ({email})")

        # Validate email
        is_valid, normalized_email = validate_email(email)

        if not is_valid:
            logger.warning(f"Invalid email format: {email}")
            error_msg = "That email doesn't look quite right. Could you please spell it out again slowly?"
            await params.result_callback(error_msg)
            return

        # Validate required fields
        if not first_name or not last_name:
            logger.warning("Missing required fields")
            error_msg = "I need both your first and last name. Could you provide those?"
            await params.result_callback(error_msg)
            return

        try:
            # Submit to Tally.so
            result = await self.tally_service.submit_appointment(
                first_name=first_name,
                last_name=last_name,
                email=normalized_email
            )

            if result["success"]:
                logger.info("✅ Appointment submitted successfully")

                # Reset booking state
                self._booking_in_progress = False

                # Return success message - LLM will then call end_conversation
                success_msg = result["message"]
                await params.result_callback(success_msg)

            else:
                logger.error(f"Appointment submission failed: {result.get('error')}")
                error_msg = result["error"]
                await params.result_callback(error_msg)
                # Keep booking in progress so user can retry
                logger.info("Booking state maintained for retry")

        except Exception as e:
            logger.error(f"Error submitting appointment: {e}")
            import traceback
            logger.error(traceback.format_exc())

            error_msg = "I encountered an error while submitting. Please try again or contact us directly."
            await params.result_callback(error_msg)

            # Reset booking state on error
            self._booking_in_progress = False

    def set_agent_transfer_callback(self, callback: Callable) -> None:
        """Register a callback for agent transfers.

        The callback receives (agent_id, persona_config) and handles
        voice switching and frontend notification.
        """
        self._on_agent_transfer = callback

    def set_a2ui_callback(self, callback: Callable) -> None:
        """Register a callback that pushes a fully-formed A2UI doc to the frontend.

        Receives one arg (the a2ui_doc dict). Used to render deterministic pharma
        cards (label citations, adverse-event reports) built from real data.
        """
        self._on_a2ui = callback

    async def _emit_a2ui(self, doc: Dict[str, Any]) -> None:
        """Best-effort deterministic A2UI push; never breaks the call if it fails."""
        if not getattr(self, "_on_a2ui", None):
            return
        try:
            await self._on_a2ui(doc)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"A2UI emit failed (non-fatal): {exc}")

    def _build_agent_roster(self, exclude_id: str = "") -> str:
        """Build the agent roster string for system prompt injection.

        Args:
            exclude_id: The current agent's ID to exclude from the roster.

        Returns:
            Formatted agent roster string.
        """
        agents = self.personas_config.get("agents", {})
        if not agents:
            return ""

        lines = []
        for agent_id, agent in agents.items():
            if agent_id == exclude_id:
                marker = " ← (THIS IS YOU)"
            else:
                marker = ""
            lines.append(
                f"  - {agent.get('name', agent_id)} (agent_id: \"{agent_id}\") — "
                f"{agent.get('role', 'Agent')}: {agent.get('description', '')}{marker}"
            )

        return self.AGENT_ROSTER_TEMPLATE.format(
            agent_count=len(agents),
            agent_list="\n".join(lines),
        )

    async def _handle_transfer_to_agent(self, params: FunctionCallParams) -> None:
        """Handle agent transfer function call.

        Switches the active agent mid-conversation: updates system prompt,
        changes TTS voice, and notifies the frontend.

        Flow:
        1. Old agent already said "Let me connect you with X" (LLM output before function call)
        2. We switch voice + system prompt and inject a handoff context message
        3. We return a directive as the function result so the NEW agent's LLM speaks
           next — it introduces itself briefly AND continues the routed task. Because
           this goes through the normal LLM->TTS path, it is queued AFTER the old
           agent's connecting line (no voice overlap) and in the new agent's voice.
        """
        import asyncio

        target_agent_id = params.arguments.get("agent_id", "").strip()
        logger.info(f"🔄 Transfer requested to agent: {target_agent_id}")

        agents = self.personas_config.get("agents", {})
        if target_agent_id not in agents:
            await params.result_callback(
                f"Sorry, I couldn't find an agent called '{target_agent_id}'. "
                f"Available agents: {', '.join(agents.keys())}"
            )
            return

        if target_agent_id == self.current_persona_id:
            await params.result_callback("You're already talking to me!")
            return

        target_persona = agents[target_agent_id]
        target_name = target_persona.get("name", target_agent_id)
        target_role = target_persona.get("role", "Agent")
        old_persona_id = self.current_persona_id
        old_agents = agents.get(old_persona_id, {})
        old_name = old_agents.get("name", old_persona_id)
        logger.info(f"🔄 Transferring from {old_name} to {target_name} ({target_agent_id})")

        # Build new system prompt with agent roster
        new_system_prompt = target_persona.get("system_prompt_override", "")
        if not new_system_prompt:
            new_system_prompt = self.llm_config.get("system_prompt", "")
        roster = self._build_agent_roster(exclude_id=target_agent_id)
        new_system_prompt = new_system_prompt.rstrip() + "\n\n" + roster

        # Extract the last user message for handoff context
        last_user_msg = ""
        if self.context and self.context.messages:
            for msg in reversed(self.context.messages):
                if msg.get("role") == "user" and msg.get("content", "").strip():
                    last_user_msg = msg["content"].strip()
                    break

        # Update the LLM context: replace system message, keep conversation history
        if self.context and self.context.messages:
            self.context.messages[0] = {"role": "system", "content": new_system_prompt}

            # Inject handoff context so the new agent knows what's happening
            handoff_note = (
                f"[HANDOFF] {old_name} transferred this call to you ({target_name}). "
                f"The user was just talking to {old_name} ({old_agents.get('role', 'Agent')}). "
            )
            if last_user_msg:
                handoff_note += f"The user's last message was: \"{last_user_msg}\". "
            handoff_note += (
                f"Pick up the conversation naturally — introduce yourself briefly and "
                f"address what they were asking about. Do NOT repeat what {old_name} already said."
            )
            self.context.messages.append({"role": "system", "content": handoff_note})
            logger.info(f"🔄 System prompt + handoff context injected for {target_name}")

        # Update current persona tracking
        self.current_persona_id = target_agent_id

        # Record the routing step and push the handoff-timeline card (Grace -> Claire
        # -> Sam, with the reason). Seed the trail with the origin agent on first hop.
        try:
            from app.services.a2ui.pharma_cards import handoff_timeline_card
            if not self._handoff_trail:
                self._handoff_trail.append({
                    "agent": old_name, "role": old_agents.get("role", ""), "reason": "Call start",
                })
            self._handoff_trail.append({
                "agent": target_name, "role": target_role,
                "reason": (last_user_msg[:70] if last_user_msg else "Routed"),
            })
            await self._emit_a2ui(handoff_timeline_card(list(self._handoff_trail)))
        except Exception as exc:  # noqa: BLE001 - visual is best-effort
            logger.error(f"handoff-timeline card failed (non-fatal): {exc}")

        # Let the outgoing agent's connecting line finish synthesizing in ITS voice
        # before we switch, so the tail of "let me connect you..." never comes out
        # in the new agent's voice.
        await asyncio.sleep(0.3)

        # Now switch the TTS voice + notify the frontend for the incoming agent.
        # The switch applies to what the new agent says next (below).
        if self._on_agent_transfer:
            await self._on_agent_transfer(target_agent_id, target_persona)

        # Hand control to the NEW agent's LLM by returning a directive as the
        # function result (not "" and not a canned TTS line). The LLM then speaks
        # with the new persona's prompt + voice, so it:
        #   - plays AFTER the outgoing connecting line (normal LLM->TTS ordering,
        #     no overlap), and
        #   - immediately continues the task it was routed for instead of asking a
        #     generic "how can I help you?".
        pickup_directive = (
            f"You are now {target_name}. Say ONE short line to greet the caller, then "
            f"immediately continue the task {old_name} routed to you"
        )
        if last_user_msg:
            pickup_directive += (
                f" — the caller was asking: \"{last_user_msg}\". Address that right now "
                f"(call call_rag_system first if you need the label to answer)."
            )
        else:
            pickup_directive += "."
        pickup_directive += (
            f" Do NOT ask what they need or say 'how can I help you' — you already know "
            f"why the call reached you. Do not repeat what {old_name} already said."
        )
        await params.result_callback(pickup_directive)
        logger.info(f"✅ Transfer to {target_name} complete — new agent continuing the task")

    def _current_rag_scope(self) -> str:
        """Resolve the RAG audience scope for the ACTIVE persona.

        Read dynamically (not cached) so a live transfer switches scope correctly:
        Claire/Alex -> "hcp" (full label), Sophie -> "patient" (patient sections).
        Returns "" for personas with no rag_scope (triage, drug_safety, trial).
        """
        if not self.rag_config.get("enabled"):
            return ""
        agent = self.personas_config.get("agents", {}).get(self.current_persona_id, {})
        return agent.get("rag_scope", "") or ""

    def _get_rag_service(self, scope: str) -> Optional[LightRAGService]:
        """Lazily build + cache a LightRAGService pointed at the scope's instance."""
        if not scope:
            return None
        if scope in self._rag_services:
            return self._rag_services[scope]

        base_url = (self.rag_config.get("scopes", {}) or {}).get(scope, "")
        if not base_url:
            logger.error(f"RAG scope '{scope}' has no configured base URL")
            return None

        service = LightRAGService(config={
            "api_url": base_url,
            "api_key": self.rag_config.get("api_key", ""),
            "mode": self.rag_config.get("mode", "mix"),
            "top_k": self.rag_config.get("top_k", 8),
            "timeout": self.rag_config.get("timeout", 20),
            # Forward token budgets — LightRAGService defaults (600/600/1000) starve
            # chunk context in mix mode and cause false "no information" refusals.
            "max_entity_tokens": self.rag_config.get("max_entity_tokens", 6000),
            "max_relation_tokens": self.rag_config.get("max_relation_tokens", 6000),
            "max_total_tokens": self.rag_config.get("max_total_tokens", 16000),
        })
        self._rag_services[scope] = service
        logger.info(f"🔎 Built LightRAG service for scope='{scope}' -> {base_url}")
        return service

    async def _handle_call_rag_system(self, params: FunctionCallParams) -> None:
        """Query the approved-label knowledge base for the ACTIVE persona's scope.

        The retrieved text is returned to the LLM (not spoken directly) so the agent
        grounds its spoken answer in it and can cite the section. A cycling "thinking
        phrase" plays automatically while this runs (call_rag_system is not skipped).
        """
        question = (params.arguments or {}).get("question", "").strip()
        scope = self._current_rag_scope()
        logger.info(f"🔎 call_rag_system (persona={self.current_persona_id}, scope='{scope}'): {question!r}")

        if not scope:
            await params.result_callback(
                "No knowledge base is available for this role. Do not answer clinical "
                "questions from general knowledge — offer to transfer the caller to the "
                "right specialist instead."
            )
            return

        if not question:
            await params.result_callback("No question was provided to look up.")
            return

        service = self._get_rag_service(scope)
        if service is None:
            await params.result_callback(
                "The knowledge base is temporarily unavailable. Offer to have medical "
                "affairs follow up rather than answering from general knowledge."
            )
            return

        # ALL queries fetch RAW label context (only_need_context) — LightRAG's
        # server-side generation is skipped entirely because the voice LLM
        # re-generates the spoken answer anyway; the duplicated generation pass
        # measured 21s at production token budgets vs ~3s retrieval-only.
        # Retrieval width comes from the shared policy: contraindications NARROW
        # (Section 4 dominant), dosing/storage WIDE, general questions moderate.
        policy = retrieval_policy(question)
        verbatim = policy["verbatim"]
        try:
            answer = await service.get_verbatim_context(
                question,
                top_k=policy["top_k"],
                chunk_top_k=policy["chunk_top_k"],
                max_total_tokens=policy["max_total_tokens"],
            )
            if not answer.strip():
                # Rare fallback: raw retrieval empty -> let LightRAG generate
                # (slow path, but better than refusing a covered question).
                logger.info(f"🔎 raw context empty, falling back to synthesized (scope='{scope}')")
                answer = await service.get_response(question)
        except Exception as exc:  # noqa: BLE001 - surface as a safe grounding message
            logger.error(f"RAG query failed (scope='{scope}'): {exc}")
            await params.result_callback(
                "The knowledge base could not be reached. Do not answer from general "
                "knowledge — offer a medical affairs follow-up instead."
            )
            return

        # Cap the context handed to the voice LLM: retrieval can return 50K+ chars,
        # which inflates prefill latency and delays the first spoken word. The
        # budget is per-policy (shared with the eval harness).
        answer = (answer or "")[:policy["context_budget"]]

        # Confidence gate: an empty answer, or LightRAG's own "no relevant context"
        # refusal, means the labeling does not cover this. Convert to an explicit
        # refusal directive instead of passing the hedge back as grounded content.
        # (Retrieval chunk counts can't gate this — LightRAG always returns top_k
        # chunks with no relevance score, so the generated answer is the real signal.)
        if not answer or not answer.strip() or _RAG_NO_CONTEXT_RE.search(answer):
            logger.info(f"🔎 RAG no-context refusal (scope='{scope}'): {answer[:80]!r}")
            await params.result_callback(
                "The approved labeling does not contain information on that. Tell the caller "
                "you can't address it from the prescribing information, and offer to have "
                "medical affairs follow up. Do NOT answer from general knowledge."
            )
            return

        logger.info(f"🔎 RAG returned {len(answer)} chars for scope='{scope}' "
                    f"(verbatim={verbatim})")

        # Deterministic visual proof of the answer. Prefer a structured dosing TABLE
        # when the retrieved text is dosing content (self-gates: returns None
        # otherwise); fall back to the plain cited-section card. Built from the
        # retrieved text itself, so the card can never disagree with what was said.
        try:
            from app.services.a2ui.pharma_cards import label_citation_card, dosing_table_card
            card = dosing_table_card(answer) or label_citation_card(answer, scope=scope)
            await self._emit_a2ui(card)
        except Exception as exc:  # noqa: BLE001 - visual is best-effort
            logger.error(f"label card failed (non-fatal): {exc}")

        # Hand the retrieved label text back to the LLM to speak, grounded + cited.
        # Every path is raw context now, so every path gets its policy directive.
        await params.result_callback(
            f"KNOWLEDGE BASE RESULT (approved Dupixent labeling, {scope} scope).{policy['directive']} "
            f"Answer ONLY from this; name the section in your spoken reply:\n\n{answer}"
        )

    async def _handle_report_adverse_event(self, params: FunctionCallParams) -> None:
        """Handle a pharmacovigilance adverse-event (AE) report.

        Called by the drug-safety agent (Sam) once all four ICH minimum-criteria
        elements have been confirmed with the caller: an identifiable reporter, an
        identifiable patient, a suspect product, and a described event. This does NOT
        end the session — Sam reads a closing line and may offer to reconnect the
        caller with a prior agent.

        For this demo the structured report is logged and appended to
        ``data/adverse_events.jsonl`` as an auditable artifact. No PII beyond what the
        caller volunteered is requested or stored.

        Args:
            params: Function call parameters carrying the structured AE fields.
        """
        args = params.arguments or {}

        # Build an immutable report record (never mutate the incoming arguments).
        report = {
            "product": (args.get("product") or "Dupixent (dupilumab)").strip(),
            "reporter_name": (args.get("reporter_name") or "").strip(),
            "reporter_contact": (args.get("reporter_contact") or "").strip(),
            "patient_descriptor": (args.get("patient_descriptor") or "").strip(),
            "dose_and_duration": (args.get("dose_and_duration") or "").strip(),
            "event_description": (args.get("event_description") or "").strip(),
            "onset": (args.get("onset") or "").strip(),
            "ongoing": bool(args.get("ongoing", False)),
            "outcome": (args.get("outcome") or "").strip(),
            "persona_id": self.current_persona_id,
        }

        # Short traceable report id (shown on-screen and stored in the audit record).
        import uuid
        report_id = f"AE-{uuid.uuid4().hex[:8].upper()}"
        logger.warning(f"🚨 ADVERSE EVENT REPORT captured ({report_id}): {report}")

        # Persist as an auditable artifact. Failure here must not break the call.
        try:
            import json
            from datetime import datetime, timezone
            from pathlib import Path

            record = {"report_id": report_id,
                      "timestamp": datetime.now(timezone.utc).isoformat(), **report}
            ae_path = Path("data/adverse_events.jsonl")
            ae_path.parent.mkdir(parents=True, exist_ok=True)
            with ae_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info(f"📝 Adverse event logged to {ae_path}")
        except Exception as exc:  # noqa: BLE001 - never let logging failure abort intake
            logger.error(f"Failed to persist adverse event report: {exc}")

        # Deterministic AE card: render the captured four-element report on screen
        # for confirmation, built from the exact fields (not LLM-guessed).
        try:
            from app.services.a2ui.pharma_cards import ae_report_card
            await self._emit_a2ui(ae_report_card(report, report_id=report_id))
        except Exception as exc:  # noqa: BLE001 - visual is best-effort
            logger.error(f"AE card failed (non-fatal): {exc}")

        # Return a short confirmation so Sam can read the closing lines from the prompt.
        # Do NOT end the session here — the safety flow closes conversationally.
        await params.result_callback(
            f"Adverse event report {report_id} recorded successfully. The safety team "
            f"will follow up."
        )

    def create_function_schemas(self) -> ToolsSchema:
        """Create function schemas for LLM tool usage.

        Returns:
            ToolsSchema containing all function definitions
        """
        end_conversation_function = FunctionSchema(
            name="end_conversation",
            description="Call this function when the caller clearly wants to end the conversation and has no further questions. Say a brief, warm goodbye first, then call this. Do NOT offer appointments or unrelated services on the medical information line.",
            properties={},
            required=[],
        )

        start_booking_function = FunctionSchema(
            name="start_appointment_booking",
            description="Start the appointment booking process. Call this when the user agrees to schedule an appointment (either after a farewell offer or mid-conversation contact request). This initiates the flow to collect their name and email.",
            properties={},
            required=[],
        )

        submit_appointment_function = FunctionSchema(
            name="submit_appointment",
            description="Submit appointment booking after collecting first name, last name, and email. CRITICAL: Only call this AFTER you have confirmed the email address character-by-character with the user and they have confirmed it is correct. Do not call this if the email has not been verbally confirmed.",
            properties={
                "first_name": {
                    "type": "string",
                    "description": "User's first name",
                },
                "last_name": {
                    "type": "string",
                    "description": "User's last name",
                },
                "email": {
                    "type": "string",
                    "description": "User's confirmed email address",
                },
            },
            required=["first_name", "last_name", "email"],
        )

        report_adverse_event_function = FunctionSchema(
            name="report_adverse_event",
            description=(
                "Log a pharmacovigilance adverse-event (side effect) report. ONLY the "
                "drug-safety agent calls this, and ONLY after confirming all four elements "
                "with the caller: an identifiable reporter, an identifiable patient, the "
                "suspect product, and a described event. Summarize the report back and get "
                "the caller's confirmation BEFORE calling this. Does not end the call."
            ),
            properties={
                "product": {
                    "type": "string",
                    "description": "Suspect product, e.g. 'Dupixent (dupilumab)'.",
                },
                "reporter_name": {
                    "type": "string",
                    "description": "Name of the person reporting (may be the patient or a caregiver/HCP).",
                },
                "reporter_contact": {
                    "type": "string",
                    "description": "Best way for the safety team to follow up (phone or email), as given.",
                },
                "patient_descriptor": {
                    "type": "string",
                    "description": "Minimal identifier for a distinct patient, e.g. 'woman, 40s'. Never full identity.",
                },
                "dose_and_duration": {
                    "type": "string",
                    "description": "Dose, how long on the drug, and time of last dose, if known.",
                },
                "event_description": {
                    "type": "string",
                    "description": "What happened, in the reporter's own words. Do not translate into medical jargon.",
                },
                "onset": {
                    "type": "string",
                    "description": "When the event started, if known.",
                },
                "ongoing": {
                    "type": "boolean",
                    "description": "True if the event is still ongoing, false if resolved/improving.",
                },
                "outcome": {
                    "type": "string",
                    "description": "Outcome so far: doctor visit, treatment stopped, hospitalization, recovering, etc.",
                },
            },
            required=["event_description"],
        )

        call_rag_system_function = FunctionSchema(
            name="call_rag_system",
            description=(
                "Query the approved Dupixent (dupilumab) prescribing-information knowledge "
                "base. Call this for EVERY clinical or product question before answering — "
                "never answer clinical questions from general knowledge. Pass the caller's "
                "question. The result is the retrieved labeling text; answer only from it and "
                "cite the section in your spoken reply."
            ),
            properties={
                "question": {
                    "type": "string",
                    "description": "The clinical/product question to look up in the approved labeling.",
                },
            },
            required=["question"],
        )

        # Agent transfer function — only add if multiple personas exist
        # call_rag_system is always registered; the handler resolves the audience scope
        # from the ACTIVE persona at call time (so live transfers pick the right workspace).
        tools_list = [
            end_conversation_function,
            start_booking_function,
            submit_appointment_function,
            report_adverse_event_function,
            call_rag_system_function,
        ]

        agents = self.personas_config.get("agents", {})
        if len(agents) > 1:
            # Build enum of valid agent IDs for the LLM
            agent_descriptions = []
            for aid, acfg in agents.items():
                agent_descriptions.append(f"{aid} ({acfg.get('name', aid)} - {acfg.get('role', '')})")

            transfer_function = FunctionSchema(
                name="transfer_to_agent",
                description=(
                    "Transfer the caller to a different agent on your team. "
                    "Call this when the user's needs are better served by another agent, "
                    "or when they ask to speak with someone specific. "
                    f"Available agents: {', '.join(agent_descriptions)}"
                ),
                properties={
                    "agent_id": {
                        "type": "string",
                        "description": "The agent_id to transfer to",
                        "enum": list(agents.keys()),
                    },
                },
                required=["agent_id"],
            )
            tools_list.append(transfer_function)

        return ToolsSchema(standard_tools=tools_list)

    def create_context(self) -> LLMContext:
        """Create the LLM context with system messages and tools.

        Uses the new universal LLMContext (replaces deprecated OpenAILLMContext).

        Returns:
            LLMContext for the conversation
        """
        tools = self.create_function_schemas()

        support_hinglish = self.language_config.get("support_hinglish", False)
        primary_language = self.language_config.get("primary", "en")

        # Get custom system prompt from config, or use default
        custom_system_prompt = self.llm_config.get("system_prompt", "")

        if custom_system_prompt:
            # Use custom system prompt from config (it already includes identity rules)
            system_message = custom_system_prompt
            # Inject agent roster so this agent knows about all other agents
            roster = self._build_agent_roster(exclude_id=self.current_persona_id)
            if roster:
                system_message = system_message.rstrip() + "\n\n" + roster
            logger.info(f"Using custom system prompt (length: {len(system_message)} chars)")
        else:
            # Default system prompt (fallback only)
            logger.warning("No custom system prompt found in config, using default")
            system_message = """
You are a helpful AI voice assistant. Keep responses SHORT and CONCISE - ideal for voice conversation.

RESPONSE RULES:
- Keep answers to 1-3 sentences maximum
- Be direct and to the point
- Speak naturally in conversational English
"""

        # No initial user prompt - greeting is handled via direct TTS
        # This prevents the LLM from generating a multi-sentence greeting
        messages = [
            {"role": "system", "content": system_message},
        ]

        # Use new universal LLMContext (replaces deprecated OpenAILLMContext)
        context = LLMContext(messages=messages, tools=tools)
        return context

    def create_context_aggregator(self) -> Any:
        """Create the context aggregator for the conversation.

        Uses LLMContextAggregatorPair (pipecat 0.0.98).
        SmartTurn v3 is configured at the transport level via turn_analyzer param.

        Returns:
            The context aggregator pair instance
        """
        if not self.llm_service:
            self.initialize_llm()

        self.context = self.create_context()  # Store for greeting access

        # Create user params (pipecat 0.0.98 - no user_turn_strategies or user_mute_strategies)
        user_params = LLMUserAggregatorParams()

        # Create assistant params (default)
        assistant_params = LLMAssistantAggregatorParams(
            expect_stripped_words=True  # TTS typically sends stripped words
        )

        # Create the universal context aggregator pair
        self.context_aggregator = LLMContextAggregatorPair(
            context=self.context,
            user_params=user_params,
            assistant_params=assistant_params
        )

        logger.info("📋 Context aggregator created (LLMContextAggregatorPair)")
        return self.context_aggregator

    def get_llm_service(self) -> LLMService:
        """Get the LLM service instance.
        
        Returns:
            The LLM service instance
        """
        if not self.llm_service:
            self.initialize_llm()
        return self.llm_service

    def get_context_aggregator(self) -> Any:
        """Get the context aggregator instance.

        Returns:
            The context aggregator instance
        """
        if not self.context_aggregator:
            self.create_context_aggregator()
        return self.context_aggregator

    def update_system_message(self, new_message: str) -> None:
        """Update the system message for the conversation.
        
        Args:
            new_message: The new system message
        """
        logger.info("Updating system message")
        # This would require recreating the context - implement as needed
        logger.warning("System message update not fully implemented")

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics.
        
        Returns:
            Dictionary containing conversation statistics
        """
        return {
            "llm_initialized": self.llm_service is not None,
            "context_aggregator_initialized": self.context_aggregator is not None,
            "tts_service_connected": self.tts_service is not None,
            "input_analyzer_ready": self.input_analyzer is not None,
        }
