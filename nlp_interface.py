from typing import Dict, Any, List, Optional, Tuple
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from quantum_agent import QuantumAIAgent
import json
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SYSTEM_PROMPT = """
You are QuantumAI, an expert in quantum computing and quantum information science.
Your role is to help users understand quantum concepts, design quantum circuits, 
and analyze quantum computation results. Be clear, precise, and educational in your responses.

When a user asks about quantum concepts or requests a quantum operation:
1. First explain the concept or operation in simple terms
2. If applicable, describe the quantum circuit that would implement it
3. If the user wants to run a quantum circuit, extract the necessary parameters
4. Provide insights into the results and their implications

Always prioritize accuracy and educational value in your responses.
"""

class QuantumNLInterface:
    def __init__(self, quantum_agent: QuantumAIAgent):
        """
        Initialize the Natural Language Interface for Quantum AI.
        
        Args:
            quantum_agent: An instance of QuantumAIAgent
        """
        load_dotenv()
        self.quantum_agent = quantum_agent
        
        # Initialize the language model
        try:
            self.llm = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-4",
                request_timeout=30
            )
            logger.info("Initialized ChatOpenAI with GPT-4")
        except Exception as e:
            logger.error(f"Failed to initialize ChatOpenAI: {str(e)}")
            raise
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", DEFAULT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        # Initialize the chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True,
            output_key="output"
        )
        
        logger.info("QuantumNLInterface initialized successfully")
    
    def process_natural_language(self, user_input: str) -> Dict[str, Any]:
        """
        Process natural language input and generate a response or quantum operation.
        
        Args:
            user_input: The user's natural language input
            
        Returns:
            Dict containing the response and any quantum results
        """
        if not user_input.strip():
            return {
                "response": "Please provide a valid input. You can ask about quantum computing concepts or request a quantum operation.",
                "status": "error"
            }
            
        logger.info(f"Processing user input: {user_input[:100]}...")
        
        try:
            # First, use the language model to understand the intent
            response = self.chain.invoke({"input": user_input})
            ai_response = response.get("output", "")
            
            if not ai_response:
                logger.warning("Empty response from language model")
                return {
                    "response": "I'm sorry, I couldn't generate a response. Please try again.",
                    "status": "error"
                }
            
            # Check if the response indicates a quantum operation should be performed
            quantum_ops = self._extract_quantum_operations(user_input, ai_response)
            
            if quantum_ops:
                try:
                    logger.info(f"Executing quantum operations: {quantum_ops}")
                    results = self._execute_quantum_operations(quantum_ops)
                    
                    if "error" in results:
                        error_msg = results.get("error", "Unknown error")
                        logger.error(f"Error in quantum execution: {error_msg}")
                        return {
                            "response": f"I tried to run a quantum circuit but encountered an error: {error_msg}",
                            "status": "error"
                        }
                    
                    # Format the quantum results for the response
                    result_summary = self._format_quantum_results(results)
                    
                    return {
                        "response": f"{ai_response}\n\n**Quantum Results:**\n{result_summary}",
                        "quantum_results": results,
                        "status": "success"
                    }
                    
                except Exception as e:
                    error_msg = f"Error executing quantum operations: {str(e)}"
                    logger.exception(error_msg)
                    return {
                        "response": "I encountered an error while trying to execute the quantum operations. Please try a different request or check your input.",
                        "status": "error"
                    }
            
            # If no quantum operations, just return the AI response
            return {
                "response": ai_response,
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"Error processing natural language: {str(e)}"
            logger.exception(error_msg)
            return {
                "response": "I'm sorry, I encountered an unexpected error while processing your request. Please try again later.",
                "status": "error"
            }
    
    def _extract_quantum_operations(self, user_input: str, llm_response: str) -> Optional[List[Dict]]:
        """
        Extract quantum operations from the user input or LLM response.
        
        This is a simplified implementation. In a production environment, you would
        want to use more sophisticated NLP techniques or fine-tune a model specifically
        for this task.
        
        Args:
            user_input: The original user input
            llm_response: The LLM's response
            
        Returns:
            List of quantum operations if any are found, None otherwise
        """
        try:
            # Convert to lowercase for case-insensitive matching
            input_lower = user_input.lower()
            
            # Check for common quantum operation patterns
            quantum_keywords = [
                "quantum circuit", "qubit", "hadamard", "h gate", "cnot", "cx",
                "quantum gate", "measure", "superposition", "entanglement",
                "quantum operation", "apply gate", "create circuit"
            ]
            
            # If no quantum-related keywords are found, return None
            if not any(keyword in input_lower for keyword in quantum_keywords):
                return None
            
            # Simple rule-based extraction (this is a placeholder)
            operations = []
            
            # Check for specific gate types
            if "hadamard" in input_lower or "h gate" in input_lower:
                # Extract qubit numbers if specified
                qubits = self._extract_qubit_numbers(input_lower, default=[0])
                operations.append({"name": "h", "qubits": qubits})
                
            if "cnot" in input_lower or "cx" in input_lower:
                # Extract control and target qubits if specified
                qubits = self._extract_qubit_numbers(input_lower, default=[0, 1])
                if len(qubits) >= 2:
                    operations.append({"name": "cx", "qubits": qubits[:2]})
            
            # If no specific operations were found but quantum keywords exist,
            # return a default operation
            if not operations and any(kw in input_lower for kw in ["quantum", "qubit"]):
                return [{"name": "h", "qubits": [0]}]
                
            return operations if operations else None
            
        except Exception as e:
            logger.error(f"Error extracting quantum operations: {str(e)}", exc_info=True)
            return None
    
    def _extract_qubit_numbers(self, text: str, default: List[int] = None) -> List[int]:
        """
        Extract qubit numbers from text using simple pattern matching.
        
        Args:
            text: Input text to search for qubit numbers
            default: Default qubit numbers to return if none are found
            
        Returns:
            List of qubit numbers
        """
        import re
        
        if default is None:
            default = [0]  # Default to qubit 0 if none specified
            
        # Look for patterns like "qubit 1", "qubits 0 and 1", "qubits 0, 1, 2", etc.
        matches = re.findall(r'(?:qubit|q)(?:\s+(?:number|#|nos?\.?|:)?\s*)(\d+(?:\s*,\s*\d+)*)', text, re.IGNORECASE)
        
        if not matches:
            return default
            
        # Extract all numbers from matches
        numbers = []
        for match in matches:
            numbers.extend([int(n) for n in re.findall(r'\d+', match)])
            
        return numbers if numbers else default
    
    def _format_quantum_results(self, results: Dict) -> str:
        """
        Format quantum results into a human-readable string.
        
        Args:
            results: Results from quantum execution
            
        Returns:
            Formatted string representation of the results
        """
        if not results or "error" in results:
            return "No results available."
            
        output = []
        
        # Add backend information if available
        backend = results.get("backend", "unknown backend")
        output.append(f"- **Backend**: {backend}")
        
        # Add job ID if available
        if "job_id" in results:
            output.append(f"- **Job ID**: {results['job_id']}")
        
        # Add counts if available
        if "counts" in results and results["counts"]:
            counts = results["counts"]
            output.append("\n**Measurement Results:**")
            
            # Sort by count (descending)
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            
            # Show top 10 results if there are many
            max_results = 10
            if len(sorted_counts) > max_results:
                output.append(f"(Showing top {max_results} results out of {len(sorted_counts)})")
                sorted_counts = sorted_counts[:max_results]
            
            # Format the counts
            for state, count in sorted_counts:
                output.append(f"  - `{state}`: {count} counts")
        
        # Add analysis if available
        if "analysis" in results and results["analysis"]:
            analysis = results["analysis"]
            output.append("\n**Analysis:**")
            if isinstance(analysis, dict) and "analysis" in analysis:
                output.append(analysis["analysis"])
            elif isinstance(analysis, str):
                output.append(analysis)
            else:
                output.append(str(analysis))
        
        return "\n".join(output)
    
    def _execute_quantum_operations(self, operations: List[Dict]) -> Dict:
        """
        Execute the extracted quantum operations.
        
        Args:
            operations: List of quantum operations to execute
            
        Returns:
            Results of the quantum operations
        """
        try:
            # Create a simple quantum circuit with the operations
            num_qubits = max([max(op.get('qubits', [0])) for op in operations] + [0]) + 1
            
            # Add measurement to the operations if not present
            has_measurement = any(op.get('name', '').lower() == 'measure' for op in operations)
            if not has_measurement:
                operations.append({"name": "measure", "qubits": list(range(num_qubits))})
            
            # Create and run the circuit
            circuit = self.quantum_agent.create_quantum_circuit(num_qubits, operations)
            optimized_circuit = self.quantum_agent.optimize_circuit(circuit)
            result = self.quantum_agent.run_circuit(optimized_circuit)
            
            if "error" in result:
                return {"error": result["error"], "status": "failed"}
                
            # Analyze and return the results
            analysis = self.quantum_agent.analyze_results(result["counts"])
            return {
                "circuit": str(optimized_circuit),
                "results": result,
                "analysis": analysis,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}


def create_gradio_interface(quantum_agent: QuantumAIAgent):
    """
    Create a Gradio interface for the Quantum AI Assistant.
    
    Args:
        quantum_agent: An instance of QuantumAIAgent
        
    Returns:
        Gradio Interface object
    """
    try:
        # Initialize the NLP interface
        nlp_interface = QuantumNLInterface(quantum_agent)
        logger.info("Initialized QuantumNLInterface for Gradio")
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: 0 auto !important;
        }
        .chatbot {
            min-height: 500px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            overflow-y: auto;
        }
        .user-message {
            background-color: #f0f7ff;
            border-radius: 15px;
            padding: 10px 15px;
            margin: 5px 0;
            max-width: 80%;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #f5f5f5;
            border-radius: 15px;
            padding: 10px 15px;
            margin: 5px 0;
            max-width: 80%;
        }
        .quantum-results {
            background-color: #f8f9fa;
            border-left: 4px solid #6c63ff;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 8px 8px 0;
        }
        .error-message {
            color: #d32f2f;
            background-color: #ffebee;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        """
        
        def process_message(message: str, chat_history: list) -> tuple:
            """Process a single message and update the chat history."""
            if not message.strip():
                return "", chat_history
                
            try:
                # Add user message to chat history
                chat_history.append((message, None))
                
                # Process the message
                response = nlp_interface.process_natural_language(message)
                
                # Format the response
                if response.get("status") == "error":
                    formatted_response = f"<div class='error-message'>{response['response']}</div>"
                else:
                    formatted_response = response["response"]
                
                # Add the response to chat history
                chat_history[-1] = (message, formatted_response)
                
                return "", chat_history
                
            except Exception as e:
                error_msg = f"An error occurred: {str(e)}"
                logger.exception(error_msg)
                chat_history[-1] = (message, f"<div class='error-message'>{error_msg}</div>")
                return "", chat_history
        
        
        # Create the Gradio interface
        with gr.Blocks(
            title="Quantum AI Research Assistant",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="indigo",
                neutral_hue="slate",
                spacing_size="sm",
                radius_size="md",
                font=["Inter", "sans-serif"]
            ),
            css=css
        ) as demo:
            # Header
            gr.Markdown("""
            # 🔬 Quantum AI Research Assistant
            Welcome! I can help you with quantum computing concepts, design quantum circuits, 
            and run simulations. Try asking me something like:
            - "Explain quantum superposition"
            - "Create a Bell state circuit"
            - "What's a quantum Fourier transform?"
            - "Run a quantum teleportation circuit"
            """)
            
            # Chat interface
            with gr.Row():
                with gr.Column(scale=3):
                    # Chatbot
                    chatbot = gr.Chatbot(
                        label="Chat",
                        show_label=False,
                        elem_id="chatbot",
                        show_copy_button=True,
                        bubble_full_width=False,
                        height=600
                    )
                    
                    # Input area
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your message",
                            placeholder="Ask me about quantum computing...",
                            container=False,
                            scale=5,
                            min_width=0,
                            max_lines=3
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=100)
                    
                    # Buttons row
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
                
                # Sidebar with information
                with gr.Column(scale=1):
                    gr.Markdown("### ℹ️ Quantum Computing Info")
                    gr.Markdown("""
                    **Available Quantum Gates:**
                    - Single-qubit: H, X, Y, Z, S, T, RX, RY, RZ
                    - Two-qubit: CNOT (CX), CY, CZ, SWAP
                    - Special: QFT (Quantum Fourier Transform)
                    
                    **Example Commands:**
                    - "Apply Hadamard to qubit 0"
                    - "Create a Bell pair between qubits 0 and 1"
                    - "Show me a 2-qubit quantum circuit"
                    - "Explain quantum entanglement"
                    """)
                    
                    # Status indicator
                    gr.Textbox(
                        label="Status",
                        value="✅ Ready to connect to quantum backend" if quantum_agent.backend else "⚠️ Using local simulator",
                        interactive=False
                    )
            
            # Example messages
            example_messages = [
                "What is quantum superposition?",
                "Create a 2-qubit Bell state circuit",
                "Explain quantum teleportation",
                "Show me a simple quantum algorithm"
            ]
            
            # Remove the show_examples function as it's not needed
            
            # Register event handlers
            msg.submit(
                process_message,
                [msg, chatbot],
                [msg, chatbot],
                queue=False
            )
            
            submit_btn.click(
                process_message,
                [msg, chatbot],
                [msg, chatbot],
                queue=False
            )
            
            clear_btn.click(
                lambda: ([], ""),
                None,
                [chatbot, msg],
                queue=False
            )
            
            # Create examples separately  
            gr.Examples(
                examples=example_messages, 
                inputs=msg, 
                label="Example Queries"
            )
            
            # Note: JavaScript functionality can be added later if needed
            
            logger.info("Created Gradio interface")
            
        return demo
        
    except Exception as e:
        logger.error(f"Error creating Gradio interface: {str(e)}", exc_info=True)
        raise

def parse_arguments():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantum AI Research Assistant")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable sharing of the Gradio interface via a public URL"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="0.0.0.0",
        help="Server name (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port (default: 7860)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Configure logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("quantum_ai_assistant.log")
            ]
        )
        
        logger.info("Starting Quantum AI Research Assistant...")
        
        # Initialize the quantum agent
        try:
            logger.info("Initializing Quantum AI Agent...")
            quantum_agent = QuantumAIAgent()
            backend_info = getattr(quantum_agent.backend, 'name', 'local simulator') if hasattr(quantum_agent, 'backend') and quantum_agent.backend else 'local simulator'
            logger.info(f"Quantum AI Agent initialized successfully. Using backend: {backend_info}")
        except Exception as e:
            logger.error(f"Failed to initialize Quantum AI Agent: {str(e)}", exc_info=True)
            raise
        
        # Create the Gradio interface
        try:
            logger.info("Creating Gradio interface...")
            demo = create_gradio_interface(quantum_agent)
            
            # Launch the interface
            logger.info("Launching Gradio interface...")
            demo.launch(
                server_name=args.server_name,
                server_port=args.server_port,
                share=args.share,
                show_error=True,
                show_api=False,
                debug=args.debug
            )
            
        except Exception as e:
            logger.error(f"Failed to launch Gradio interface: {str(e)}", exc_info=True)
            raise
            
    except KeyboardInterrupt:
        logger.info("Shutting down Quantum AI Research Assistant...")
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)
        print(f"\nA critical error occurred. Please check the logs at 'quantum_ai_assistant.log' for details.")
        print(f"Error: {str(e)}")
        exit(1)
