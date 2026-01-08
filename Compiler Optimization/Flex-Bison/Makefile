CC=gcc
CFLAGS=-Wall -g
FLEX=flex
BISON=bison
LDFLAGS=-lm

# Source files
LEX_SRC=finalAssignment.l
YACC_SRC=finalAssignment.y
C_SRCS=tensor_graph.c optimizer.c cuda_gen.c geometry.c llvm_lowering.c validator.c
HDRS=types.h tensor_graph.h optimizer.h cuda_gen.h geometry.h llvm_lowering.h validator.h

# Generated files
LEX_OUT=lex.yy.c
YACC_OUT=finalAssignment.tab.c
YACC_HEADER=finalAssignment.tab.h

# Object files
OBJS=$(LEX_OUT:.c=.o) $(YACC_OUT:.c=.o) $(C_SRCS:.c=.o)

# Target executable
TARGET=compiler

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

$(LEX_OUT): $(LEX_SRC) $(YACC_HEADER)
	$(FLEX) $(LEX_SRC)

$(YACC_OUT) $(YACC_HEADER): $(YACC_SRC)
	$(BISON) -d $(YACC_SRC)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) $(LEX_OUT) $(YACC_OUT) $(YACC_HEADER) generated_kernels.cu
	rm -f *.ll *.bc llvm_output/*.ll llvm_output/*.txt llvm_output/*.log
	rm -rf llvm_output

test: $(TARGET)
	@echo "Testing with simple_test.txt:"
	./$(TARGET) < tests/simple_test.txt

# LLVM Integration targets
llvm-test: $(TARGET)
	@echo "========================================="
	@echo "LLVM Integration Test"
	@echo "========================================="
	@mkdir -p llvm_output
	@echo "Generating LLVM IR from test case..."
	./$(TARGET) < tests/simple_test.txt
	@if [ -f tensor_output.ll ]; then \
		echo "✓ LLVM IR generated: tensor_output.ll"; \
		echo "Running LLVM analysis pipeline..."; \
		./llvm_analyze.sh tensor_output.ll llvm_output; \
	else \
		echo "✗ No LLVM IR generated"; \
	fi

llvm-analyze: tensor_output.ll
	@echo "Running LLVM analysis on tensor_output.ll..."
	./llvm_analyze.sh tensor_output.ll llvm_output

.PHONY: llvm-test llvm-analyze