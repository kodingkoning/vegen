#ifndef SOLVER_H
#define SOLVER_H

#include "Packer.h"
#include "VectorPackContext.h"
#include "VectorPackSet.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include <bitset>

// A lightweight wrapper around VectorPack::OperandPack
// to track resolved lanes (lanes with known producers)
struct UnresolvedOperandPack {
  using BitsetTy = std::bitset<64>;

  const VectorPack::OperandPack *Pack;
  BitsetTy ResolvedLanes;

  UnresolvedOperandPack(UnresolvedOperandPack &&) = default;
  UnresolvedOperandPack &operator=(UnresolvedOperandPack &&) = default;

  UnresolvedOperandPack(const VectorPack::OperandPack &Pack) : Pack(&Pack) {
    for (unsigned i = 0; i < Pack.size(); i++) {
      if (llvm::isa<llvm::Constant>(Pack[i]))
        ResolvedLanes.set(i);
    }
  }

  void resolveOneLane(unsigned i) { ResolvedLanes.set(i); }
  bool resolved() const { return ResolvedLanes.all(); }
  bool operator<(const UnresolvedOperandPack &Other) const {
    auto RL = ResolvedLanes.to_ullong();
    auto RL2 = ResolvedLanes.to_ullong();
    return std::tie(Pack, RL) < std::tie(Other.Pack, RL2);
  }

  bool operator==(const UnresolvedOperandPack &Other) const {
    return Pack == Other.Pack && ResolvedLanes == Other.ResolvedLanes;
  }
};

class PackEnumerationCache {
  // Mapping a focus instruction to the set of packs it can involve in,
  // Assumming every instruction before it is free and below not.
  llvm::DenseMap<llvm::Instruction *, std::vector<const VectorPack *>>
      FocusToPacksMap;

public:
  llvm::ArrayRef<const VectorPack *> getPacks(llvm::Instruction *I,
                                              bool &InCache) const {
    auto It = FocusToPacksMap.find(I);
    if (It != FocusToPacksMap.end()) {
      InCache = true;
      return It->second;
    }
    InCache = false;
    return {};
  }

  void insert(llvm::Instruction *I, std::vector<const VectorPack *> &&Packs) {
    FocusToPacksMap[I] = Packs;
  }
};

class MatchManager;
class Frontier {
  friend struct FrontierHashInfo;
  llvm::BasicBlock *BB;
  const VectorPackContext *VPCtx;
  llvm::BasicBlock::reverse_iterator BBIt;
  std::vector<UnresolvedOperandPack> UnresolvedPacks;
  llvm::BitVector UnresolvedScalars;
  // Instructions we haven't assigned yet.
  llvm::BitVector FreeInsts;

  void freezeOneInst(unsigned);
  bool resolveOperandPack(const VectorPack &VP, UnresolvedOperandPack &UP);
  void advanceBBIt();
  bool isFreeInst(llvm::Instruction *I) {
    return FreeInsts[VPCtx->getScalarId(I)];
  }

  // Remove any packs that use frozen instructions.
  std::vector<const VectorPack *>
      filterFrozenPacks(llvm::ArrayRef<const VectorPack *>) const;

public:
  // Create the initial frontier, which surrounds the whole basic block
  Frontier(llvm::BasicBlock *BB, const VectorPackContext *VPCtx);
  std::unique_ptr<Frontier> advance(const VectorPack *, float &Cost,
                                    llvm::TargetTransformInfo *TTI) const;
  std::unique_ptr<Frontier> advance(llvm::Instruction *, float &Cost,
                                    llvm::TargetTransformInfo *TTI) const;
  llvm::Instruction *getNextFreeInst() const;
  std::vector<const VectorPack *>
  nextAvailablePacks(Packer *, PackEnumerationCache *) const;
};

// Hashing support for `Frontier`
struct FrontierHashInfo {
  static Frontier *getEmptyKey() { return nullptr; }

  static Frontier *getTombstoneKey() {
    return reinterpret_cast<Frontier *>(1);
  }

  static unsigned getHashValue(const Frontier *Frt) {
    using namespace llvm;

    if (Frt == getEmptyKey()) {
      return hash_combine(reinterpret_cast<BasicBlock *>(0),
                          ArrayRef<uint64_t>(), ArrayRef<uint64_t>(),
                          ArrayRef<uint64_t>());
    } else if (Frt == getTombstoneKey()) {
      return hash_combine(reinterpret_cast<BasicBlock *>(1),
                          ArrayRef<uint64_t>(), ArrayRef<uint64_t>(),
                          ArrayRef<uint64_t>());
    }

    // Interpret unresolvedOperandPack as a uint64_t array...
    ArrayRef<uint64_t> UPRaw(
        reinterpret_cast<const uint64_t *>(Frt->UnresolvedPacks.data()),
        Frt->UnresolvedPacks.size() * 2);

    return hash_combine(Frt->BB, Frt->UnresolvedScalars.getData(),
                        Frt->FreeInsts.getData(), UPRaw);
  }

  static bool isTombstoneOrEmpty(const Frontier *Frt) {
    return Frt == getTombstoneKey() || Frt == getEmptyKey();
  }

  static bool isEqual(const Frontier *A, const Frontier *B) {
    if (isTombstoneOrEmpty(A) || isTombstoneOrEmpty(B))
      return A == B;

    return A->BB == B->BB && A->UnresolvedScalars == B->UnresolvedScalars &&
           A->FreeInsts == B->FreeInsts &&
           A->UnresolvedPacks == B->UnresolvedPacks;
  }
};

class UCTNode;
class UCTNodeFactory {
  std::vector<std::unique_ptr<Frontier>> Frontiers;
  std::vector<std::unique_ptr<UCTNode>> Nodes;
  llvm::DenseMap<Frontier *, UCTNode *, FrontierHashInfo> FrontierToNodeMap;

public:
  UCTNode *getNode(std::unique_ptr<Frontier> Frontier);
};

class UCTNode {
public:
  // The next action state pair
  struct OutEdge {
    const VectorPack *VP; // NULL means we choose Frt->getNextFreeInst();
    UCTNode *Next;
    float Cost; // Reward
  };

private:
  friend class UCTNodeFactory;

  // State
  const Frontier *Frt;
  // Return
  float TotalCost;
  uint64_t Count;

  std::vector<OutEdge> OutEdges;

  UCTNode(const Frontier *Frt) : Frt(Frt), TotalCost(0), Count(0) {}

public:
  // Fill out the out edge
  void expand(UCTNodeFactory *Factory, Packer *Packer,
              PackEnumerationCache *EnumCache, llvm::TargetTransformInfo *);
  bool expanded() { return !OutEdges.empty() && !isTerminal(); }
  bool isTerminal() const { return !Frt->getNextFreeInst(); }
  llvm::ArrayRef<OutEdge> next() const { return OutEdges; }
  // The classic UCT score
  float score(unsigned ParentCount, float C) const {
    return (-TotalCost / float(Count)) +
           C * sqrtf(logf(ParentCount) / float(Count));
  }
  uint64_t visitCount() const { return Count; }
  void update(float Cost) {
    TotalCost += Cost;
    Count++;
  }
};

class UCTSearch {
  // The exploration factor in UCT
  float C;
  UCTNodeFactory *Factory;
  Packer *Packer;
  PackEnumerationCache EnumCache;
  llvm::TargetTransformInfo *TTI;

public:
  UCTSearch(float C, UCTNodeFactory *Factory, class Packer *Packer,
            llvm::TargetTransformInfo *TTI)
      : C(C), Factory(Factory), Packer(Packer), TTI(TTI) {}
  // Run MCTS for some iterations
  void run(UCTNode *Root, unsigned Iter);
  // E.g., value function or rollout
  virtual float evalLeafNode(UCTNode *) { return 0; }
};

#endif