#include "ConsecutiveCheck.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "TestConsecutiveCheck"

using namespace llvm;

namespace llvm {
void initializeTestConsecutiveCheckPass(PassRegistry &);
}

namespace {
struct TestConsecutiveCheck : public FunctionPass {
  static char ID;

  TestConsecutiveCheck() : FunctionPass(ID) {
    initializeTestConsecutiveCheckPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ScalarEvolutionWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
  }

  bool runOnFunction(Function &) override;
};

} // namespace

char TestConsecutiveCheck::ID = 0;

bool TestConsecutiveCheck::runOnFunction(Function &F) {
  auto &SE = getAnalysis<ScalarEvolutionWrapperPass>().getSE();
  auto &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();

  const DataLayout &DL = F.getParent()->getDataLayout();
  for (auto &I : instructions(&F))
    for (auto &J : instructions(&F))
      if (isConsecutive(&I, &J, DL, SE, LI)) {
        LLVM_DEBUG(errs() << I << " and " << J << " are consecutive\n");
      }
  return false;
}

INITIALIZE_PASS_BEGIN(TestConsecutiveCheck, "test-consecutive-check",
                      "Test Consecutive Access Check", false, false)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(TestConsecutiveCheck, "test-consecutive-check",
                    "Test Consecutive Access Check", false, false)

static struct RegisterMe {
  RegisterMe() {
    initializeTestConsecutiveCheckPass(*PassRegistry::getPassRegistry());
  }
} X;
