from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import inflect


def json_parser(num_hints):
    num_hints = int(num_hints)
    hints = []
    for i in range(1, num_hints + 1):
        name = f"Hint{i}"
        description = f"The {inflect.engine().ordinal(i)} hint"
        hints.append(ResponseSchema(name=name, description=description))

    output_parser = StructuredOutputParser.from_response_schemas(hints)
    return output_parser


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)