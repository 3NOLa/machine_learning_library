#include "tokenizer.h"


char** tokeknize(const char* text, int* token_count)
{
	char** tokenizer = NULL;

	char* context = NULL;
	char* token = strtok_s(text, " ", &context);
	while (token)
	{
		char* t = remove_punctuation(token);
		to_lowercase(t);

		char** tokenizer_check = (char**)realloc(tokenizer, sizeof(char*) * (*token_count + 1));
		if (!tokenizer_check) {
			fprintf(stderr, "Erorr: alocetying memory for tokenizer in tokeknize\n");
			return NULL;
		}

		tokenizer = tokenizer_check;
		tokenizer[*token_count++] = t;

		token = strtok_s(NULL, " ", &context);
	}

	return tokenizer;
}

void to_lowercase(char* str)
{
	while (*str)
		*str = tolower(*str++);
}

char* remove_punctuation(const char* input)
{
	int size = strlen(input);
	char* str = (char*)malloc(sizeof(char)* (size + 1));
	int j = 0;

	for (int i = 0; i < size; i++)
		if (isdigit(input[i]))
			str[j++] = input[i];

	str[j] = '\0';
	return str;
}