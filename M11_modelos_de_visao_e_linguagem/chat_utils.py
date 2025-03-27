
class Chat:
    """Classe para interagir com um modelo de geração de texto."""

    def __init__(self, model, processor):
        """

        Parâmetros
        ----------
        model
            modelo a ser usado
        processor
            processador de imagem e texto
        """

        self.model = model
        self.processor = processor
        self.messages = []

    def add_to_chat(self, message, image=None, role="user"):
        """Adiciona uma mensagem ao histórico da conversa.
        
        Parâmetros
        ----------
        message
            Mensagem a ser adicionada
        image
            Imagem a ser adicionada
        role
            Quem enviou a mensagem, pode ser "user" ou "assistant"
        """

        content = []
        if image is not None:
            content.append({
                "type": "image",
            })
        content.append({
            "type": "text",
            "text": message
        })
        
        self.messages.append({
            "role": role,
            "content": content
        })

    def clear_history(self):
        """Limpa o histórico da conversa."""
        self.messages = []

    def query_model(self, message, image=None, max_new_tokens=128, return_inputs=False):
        """Envia uma mensagem ao modelo e retorna a resposta.
        
        Parâmetros
        ----------
        message
            Mensagem a ser enviada
        image
            Imagem a ser enviada
        max_new_tokens
            Número máximo de tokens a serem gerados
        return_inputs
            Se True, retorna as entradas do modelo
        """

        processor = self.processor

        self.add_to_chat(message, image=image)

        text = processor.apply_chat_template(
            self.messages, tokenize=False, add_generation_prompt=True
        )

        if image is None:
            images = None
        else:
            images = [image]

        inputs = processor(
            text=[text],
            images=images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        generated_ids = self.model.generate(**inputs.to("cuda"), max_new_tokens=max_new_tokens)
        num_input_tokens = len(inputs["input_ids"][0])
        output_text = processor.batch_decode(
            generated_ids[:, num_input_tokens:], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        output = output_text
        if return_inputs:
            output = (output, inputs)

        return output

