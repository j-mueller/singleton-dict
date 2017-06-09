{-# LANGUAGE TypeOperators, PolyKinds, DataKinds, KindSignatures, ConstraintKinds #-}
{-# LANGUAGE TypeFamilies, UndecidableInstances, TemplateHaskell, TypeInType #-}
{-# LANGUAGE FlexibleInstances, GADTs, FlexibleContexts, ScopedTypeVariables #-}
{-# LANGUAGE Trustworthy #-}
{-# LANGUAGE Strict #-}
module Data.Type.Dict
  ( Dict, Dict'(..), SDict', SizeProxy, SizeProxySym0
  , natProxy, sNatProxy, NatProxy, NatProxySym0
  , dictNull, sDictNull, DictNull, DictNullSym0, DictNullSym1
  , size, sSize, Size, SizeSym0, SizeSym1
  , dictLookup, sDictLookup, DictLookup, DictLookupSym0, DictLookupSym1, DictLookupSym2
  , (!?), (:!?), (%:!?), (:!?$), (:!?$$), (:!?$$$)
  , empty, sEmpty, Empty, EmptySym0, EmptySym1
  , singleton, sSingleton, Singleton, SingletonSym0, SingletonSym1, SingletonSym2
  , insert, sInsert, Insert, InsertSym0, InsertSym1, InsertSym2, InsertSym3
  , insertWith, sInsertWith, InsertWith, InsertWithSym0, InsertWithSym1, InsertWithSym2, InsertWithSym3, InsertWithSym4
  , insertLookupWithKey, sInsertLookupWithKey, InsertLookupWithKey, InsertLookupWithKeySym0, InsertLookupWithKeySym1, InsertLookupWithKeySym2, InsertLookupWithKeySym3, InsertLookupWithKeySym4
  , deleteFindMin, sDeleteFindMin, DeleteFindMin, DeleteFindMinSym0, DeleteFindMinSym1
  , deleteFindMax, sDeleteFindMax, DeleteFindMax, DeleteFindMaxSym0, DeleteFindMaxSym1
  , delete, sDelete, Delete, DeleteSym0, DeleteSym1, DeleteSym2
  , adjust, sAdjust, Adjust, AdjustSym0, AdjustSym1, AdjustSym2, AdjustSym3
  , adjustWithKey, sAdjustWithKey, AdjustWithKey, AdjustWithKeySym0, AdjustWithKeySym1, AdjustWithKeySym2, AdjustWithKeySym3
  , update, sUpdate, Update, UpdateSym0, UpdateSym1, UpdateSym2, UpdateSym3
  , updateWithKey, sUpdateWithKey, UpdateWithKey, UpdateWithKeySym0, UpdateWithKeySym1, UpdateWithKeySym2, UpdateWithKeySym3
  , updateLookupWithKey, sUpdateLookupWithKey, UpdateLookupWithKey, UpdateLookupWithKeySym0, UpdateLookupWithKeySym1, UpdateLookupWithKeySym2, UpdateLookupWithKeySym3
  , trim, sTrim, Trim, TrimSym0, TrimSym1, TrimSym2, TrimSym3
  , trimLookupLo, sTrimLookupLo, TrimLookupLo, TrimLookupLoSym0, TrimLookupLoSym1, TrimLookupLoSym2, TrimLookupLoSym3
  , filterGt, sFilterGt, FilterGt, FilterGtSym0, FilterGtSym1, FilterGtSym2
  , filterLt, sFilterLt, FilterLt, FilterLtSym0, FilterLtSym1, FilterLtSym2
  , split, sSplit, Split, SplitSym0, SplitSym1, SplitSym2
  , splitLookup, sSplitLookup, SplitLookup, SplitLookupSym0, SplitLookupSym1, SplitLookupSym2
  , unions, sUnions, Unions, UnionsSym0, UnionsSym1, UnionsSym2
  , union, sUnion, Union, UnionSym0, UnionSym1, UnionSym2
  , unionWith, sUnionWith, UnionWith, UnionWithSym0, UnionWithSym1, UnionWithSym2, UnionWithSym3
  , unionWithKey, sUnionWithKey, UnionWithKey, UnionWithKeySym0, UnionWithKeySym1, UnionWithKeySym2, UnionWithKeySym3
  , difference, sDifference, Difference, DifferenceSym0, DifferenceSym1, DifferenceSym2
  , intersection, sIntersection, Intersection, IntersectionSym0, IntersectionSym1, IntersectionSym2
  , intersectionWith, sIntersectionWith, IntersectionWith, IntersectionWithSym0, IntersectionWithSym1, IntersectionWithSym2, IntersectionWithSym3
  , intersectionWithKey, sIntersectionWithKey, IntersectionWithKey, IntersectionWithKeySym0, IntersectionWithKeySym1, IntersectionWithKeySym2, IntersectionWithKeySym3
  , fold, sFold, Fold, FoldSym0, FoldSym1, FoldSym2, FoldSym3
  , foldWithKey, sFoldWithKey, FoldWithKey, FoldWithKeySym0, FoldWithKeySym1, FoldWithKeySym2, FoldWithKeySym3
  , foldL, sFoldL, FoldL, FoldLSym0, FoldLSym1, FoldLSym2, FoldLSym3
  , foldR, sFoldR, FoldR, FoldRSym0, FoldRSym1, FoldRSym2, FoldRSym3
  , assocs, sAssocs, Assocs, AssocsSym0, AssocsSym1
  , elems, sElems, Elems, ElemsSym0, ElemsSym1
  , keys, sKeys, Keys, KeysSym0, KeysSym1
  , fromList, sFromList, FromList, FromListSym0, FromListSym1, FromListSym2
  , fromListWith, sFromListWith, FromListWith, FromListWithSym0, FromListWithSym1, FromListWithSym2, FromListWithSym3
  , fromListWithKey, sFromListWithKey, FromListWithKey, FromListWithKeySym0, FromListWithKeySym1, FromListWithKeySym2, FromListWithKeySym3
  , toList, sToList, ToList, ToListSym0, ToListSym1
  , toAscList, sToAscList, ToAscList, ToAscListSym0, ToAscListSym1
  , toDescList, sToDescList, ToDescList, ToDescListSym0, ToDescListSym1
  , dictMap, sDictMap, DictMap, DictMapSym0, DictMapSym1, DictMapSym2
  , mapWithKey, sMapWithKey, MapWithKey, MapWithKeySym0, MapWithKeySym1, MapWithKeySym2
  , mapAccum, sMapAccum, MapAccum, MapAccumSym0, MapAccumSym1, MapAccumSym2, MapAccumSym3
  , mapAccumWithKey, sMapAccumWithKey, MapAccumWithKey, MapAccumWithKeySym0, MapAccumWithKeySym1, MapAccumWithKeySym2, MapAccumWithKeySym3
  , mapAccumL, sMapAccumL, MapAccumL, MapAccumLSym0, MapAccumLSym1, MapAccumLSym2, MapAccumLSym3
  , mapAccumR, sMapAccumR, MapAccumR, MapAccumRSym0, MapAccumRSym1, MapAccumRSym2, MapAccumRSym3
  , dictFilter, sDictFilter, DictFilter, DictFilterSym0, DictFilterSym1, DictFilterSym2
  , filterWithKey, sFilterWithKey, FilterWithKey, FilterWithKeySym0, FilterWithKeySym1, FilterWithKeySym2
  , partition, sPartition, Partition, PartitionSym0, PartitionSym1, PartitionSym2
  , partitionWithKey, sPartitionWithKey, PartitionWithKey, PartitionWithKeySym0, PartitionWithKeySym1, PartitionWithKeySym2
 ) where

import Data.Kind
import Data.Monoid
import Data.Proxy
import Data.Singletons
import Data.Singletons.Prelude
import Data.Singletons.TH
import GHC.TypeLits


$(singletons [d|
  data Dict' s k a  =
      Tip
    | Bin s k a (Dict' s k a) (Dict' s k a)

  data SizeProxy (a :: Type) = SizeProxy
 |])

$(singletons [d|
  natProxy :: SizeProxy Nat
  natProxy = SizeProxy

  dictNull :: Dict' s k a -> Bool
  dictNull t = case t of
    Tip             -> True
    Bin sz k x l r  -> False

  size :: Num s => Dict' s k a -> s
  size t = case t of
    Tip             -> 0
    Bin sz k x l r  -> sz

  dictLookup :: Ord k => k -> Dict' s k a -> Maybe a
  dictLookup k t = case t of
    Tip -> Nothing
    Bin sz kx x l r -> case compare k kx of
      LT -> dictLookup k l
      GT -> dictLookup k r
      EQ -> Just x

  (!?) :: Ord k => Dict' s k a -> k -> Maybe a
  t !? k = dictLookup k t

  empty :: SizeProxy s -> Dict' s k a
  empty _ = Tip

  singleton :: Num s => k -> a -> Dict' s k a
  singleton k x = Bin 1 k x Tip Tip

  delta :: Num s => s
  delta = 5

  ratio :: Num s => s
  ratio = 2

  balance :: (Num s, Ord s) => k -> a -> Dict' s k a -> Dict' s k a -> Dict' s k a
  balance k x l r
    | sizeL + sizeR <= 1    = Bin sizeX k x l r
    | sizeR >= delta*sizeL  = rotateL k x l r
    | sizeL >= delta*sizeR  = rotateR k x l r
    | otherwise             = Bin sizeX k x l r
    where
      sizeL = size l
      sizeR = size r
      sizeX = sizeL + sizeR + 1

  rotateL :: (Ord s, Num s) => k -> a -> Dict' s k a -> Dict' s k a -> Dict' s k a
  rotateL k x l r@(Bin _ _ _ ly ry)
    | size ly < ratio * size ry = singleL k x l r
    | otherwise                         = doubleL k x l r

  rotateR :: (Ord s, Num s) => k -> a -> Dict' s k a -> Dict' s k a -> Dict' s k a
  rotateR k x l@(Bin _ _ _ ly ry) r
    | size ry < ratio * size ly = singleR k x l r
    | otherwise                         = doubleR k x l r

  singleL :: Num s => k -> a -> Dict' s k a -> Dict' s k a -> Dict' s k a
  singleL k1 x1 t1 (Bin _ k2 x2 t2 t3)  = bin' k2 x2 (bin' k1 x1 t1 t2) t3

  singleR :: Num s => k -> a -> Dict' s k a -> Dict' s k a -> Dict' s k a
  singleR k1 x1 (Bin _ k2 x2 t1 t2) t3  = bin' k2 x2 t1 (bin' k1 x1 t2 t3)

  doubleL :: Num s => k -> a -> Dict' s k a -> Dict' s k a -> Dict' s k a
  doubleL k1 x1 t1 (Bin _ k2 x2 (Bin _ k3 x3 t2 t3) t4) = bin' k3 x3 (bin' k1 x1 t1 t2) (bin' k2 x2 t3 t4)

  doubleR :: Num s => k -> a -> Dict' s k a -> Dict' s k a -> Dict' s k a
  doubleR k1 x1 (Bin _ k2 x2 t1 (Bin _ k3 x3 t2 t3)) t4 = bin' k3 x3 (bin' k2 x2 t1 t2) (bin' k1 x1 t3 t4)

  bin' :: Num s => k -> a -> Dict' s k a -> Dict' s k a -> Dict' s k a
  bin' k x l r
    = Bin (size l + size r + 1) k x l r

  insert :: (Num s, Ord s, Ord k) => k -> a -> Dict' s k a -> Dict' s k a
  insert kx x t = case t of
    Tip -> singleton kx x
    Bin sz ky y l r -> case compare kx ky of
      LT -> balance ky y (insert kx x l) r
      GT -> balance ky y l (insert kx x r)
      EQ -> Bin sz kx x l r

  insertWith :: (Num s, Ord s, Ord k) => (a -> a -> a) -> k -> a -> Dict' s k a -> Dict' s k a
  insertWith f k x m = insertWithKey (\k x y -> f x y) k x m

  insertWithKey :: (Num s, Ord s, Ord k) => (k -> a -> a -> a) -> k -> a -> Dict' s k a -> Dict' s k a
  insertWithKey f kx x t = case t of
    Tip -> singleton kx x
    Bin sy ky y l r -> case compare kx ky of
      LT -> balance ky y (insertWithKey f kx x l) r
      GT -> balance ky y l (insertWithKey f kx x r)
      EQ -> Bin sy ky (f ky x y) l r

  insertLookupWithKey :: (Num s, Ord s, Ord k) => (k -> a -> a -> a) -> k -> a -> Dict' s k a -> (Maybe a, Dict' s k a)
  insertLookupWithKey f kx x t = case t of
    Tip -> (Nothing, singleton kx x)
    Bin sy ky y l r -> case compare kx ky of
      LT -> let (found,l') = insertLookupWithKey f kx x l in (found, balance ky y l' r)
      GT -> let (found,r') = insertLookupWithKey f kx x r in (found, balance ky y l r')
      EQ -> (Just y, Bin sy ky (f ky x y) l r)

  deleteFindMin :: (Num s, Ord s) => Dict' s k a -> ((k, a), Dict' s k a)
  deleteFindMin t = case t of
    Bin _ k x l r -> case l of
      Tip             -> ((k, x), r)
      Bin _ _ _ _ _  -> case deleteFindMin l of
        (km, l') -> (km, balance k x l' r)
    Tip             -> (error "deleteFindMin: can not return the minimal element of an empty map", Tip)

  deleteFindMax :: (Num s, Ord s) => Dict' s k a -> ((k, a), Dict' s k a)
  deleteFindMax t = case t of
    Bin _ k x l r -> case r of
      Tip             -> ((k, x), l)
      Bin _ _ _ _ _  -> case deleteFindMax r of
        (km, r') -> (km, balance k x l r')
    Tip             -> (error "deleteFindMax: can not return the maximal element of an empty map", Tip)

  glue :: (Num s, Ord s) => Dict' s k a -> Dict' s k a -> Dict' s k a
  glue l r =
    case l of
      Tip -> r
      Bin _ _ _ _ _ -> case r of
        Tip -> l
        Bin _ _ _ _ _ ->
          if size l > size r
          then let ((km, m), l') = deleteFindMax l in balance km m l' r
          else let ((km, m), r') = deleteFindMin r in balance km m l r'

  delete :: (Num s, Ord s, Ord s, Ord k) => k -> Dict' s k a -> Dict' s k a
  delete k t = case t of
    Tip -> Tip
    Bin sx kx x l r -> case compare k kx of
      LT -> balance kx x (delete k l) r
      GT -> balance kx x l (delete k r)
      EQ -> glue l r

  insertMax :: (Num s, Ord s) => k -> a -> Dict' s k a -> Dict' s k a
  insertMax kx x t = case t of
    Tip             -> singleton kx x
    Bin sz ky y l r -> balance ky y l (insertMax kx x r)

  insertMin :: (Num s, Ord s) => k -> a -> Dict' s k a -> Dict' s k a
  insertMin kx x t = case t of
    Tip             -> singleton kx x
    Bin sz ky y l r -> balance ky y (insertMin kx x l) r

  join :: (Num s, Ord s, Ord k) => k -> a -> Dict' s k a -> Dict' s k a -> Dict' s k a
  join kx x l r = case l of
    Tip -> insertMin kx x r
    Bin sizeL ky y ly ry -> case r of
      Tip -> insertMax kx x l
      Bin sizeR kz z lz rz
        | delta*sizeL <= sizeR -> balance kz z (join kx x l lz) rz
        | delta*sizeR <= sizeL -> balance ky y ly (join kx x ry r)
        | otherwise            -> bin' kx x l r

  merge :: (Num s, Ord s) => Dict' s k a -> Dict' s k a -> Dict' s k a
  merge l r = case l of
    Tip -> r
    Bin sizeL kx x lx rx -> case r of
      Tip -> l
      Bin sizeR ky y ly ry
        | delta*sizeL <= sizeR -> balance ky y (merge l ly) ry
        | delta*sizeR <= sizeL -> balance kx x lx (merge rx r)
        | otherwise            -> glue l r

  adjust :: (Num s, Ord s, Ord k) => (a -> a) -> k -> Dict' s k a -> Dict' s k a
  adjust f k m = adjustWithKey (\k x -> f x) k m

  adjustWithKey :: (Num s, Ord s, Ord k) => (k -> a -> a) -> k -> Dict' s k a -> Dict' s k a
  adjustWithKey f k m = updateWithKey (\k x -> Just (f k x)) k m

  update :: (Num s, Ord s, Ord k) => (a -> Maybe a) -> k -> Dict' s k a -> Dict' s k a
  update f k m = updateWithKey (\k x -> f x) k m

  updateWithKey :: (Num s, Ord s, Ord k) => (k -> a -> Maybe a) -> k -> Dict' s k a -> Dict' s k a
  updateWithKey f k t = case t of
    Tip -> Tip
    Bin sx kx x l r -> case compare k kx of
      LT -> balance kx x (updateWithKey f k l) r
      GT -> balance kx x l (updateWithKey f k r)
      EQ -> case f kx x of
        Just x' -> Bin sx kx x' l r
        Nothing -> glue l r

  updateLookupWithKey :: (Num s, Ord s, Ord k) => (k -> a -> Maybe a) -> k -> Dict' s k a -> (Maybe a, Dict' s k a)
  updateLookupWithKey f k t = case t of
    Tip -> (Nothing, Tip)
    Bin sx kx x l r -> case compare k kx of
      LT -> let (found, l') = updateLookupWithKey f k l in (found, balance kx x l' r)
      GT -> let (found, r') = updateLookupWithKey f k r in (found, balance kx x l r')
      EQ -> case f kx x of
        Just x' -> (Just x', Bin sx kx x' l r)
        Nothing -> (Just x, glue l r)

  trim :: (k -> Ordering) -> (k -> Ordering) -> Dict' s k a -> Dict' s k a
  trim cmplo cmphi Tip = Tip
  trim cmplo cmphi t@(Bin sx kx x l r) = case cmplo kx of
    LT -> case cmphi kx of
      GT -> t
      LT -> trim cmplo cmphi l
      EQ -> trim cmplo cmphi l
    GT -> trim cmplo cmphi r
    EQ -> trim cmplo cmphi r

  trimLookupLo :: (Num s, Ord s, Ord k) => k -> (k -> Ordering) -> Dict' s k a -> (Maybe a, Dict' s k a)
  trimLookupLo lo cmphi Tip = (Nothing, Tip)
  trimLookupLo lo cmphi t@(Bin sx kx x l r) = case compare lo kx of
    LT -> case cmphi kx of
      GT -> (dictLookup lo t, t)
      LT -> trimLookupLo lo cmphi l
      EQ -> trimLookupLo lo cmphi l
    GT -> trimLookupLo lo cmphi r
    EQ -> (Just x, trim (compare lo) cmphi r)

  filterGt :: (Num s, Ord s, Ord k) => (k -> Ordering) -> Dict' s k a -> Dict' s k a
  filterGt cmp Tip = Tip
  filterGt cmp (Bin sx kx x l r) = case cmp kx of
    LT -> join kx x (filterGt cmp l) r
    GT -> filterGt cmp r
    EQ -> r

  filterLt :: (Num s, Ord s, Ord k) => (k -> Ordering) -> Dict' s k a -> Dict' s k a
  filterLt cmp Tip = Tip
  filterLt cmp (Bin sx kx x l r)
    = case cmp kx of
        LT -> filterLt cmp l
        GT -> join kx x l (filterLt cmp r)
        EQ -> l

  split :: (Num s, Ord s, Ord k) => k -> Dict' s k a -> (Dict' s k a, Dict' s k a)
  split k Tip = (Tip, Tip)
  split k (Bin sx kx x l r) = case compare k kx of
    LT -> let (lt,gt) = split k l in (lt, join kx x gt r)
    GT -> let (lt,gt) = split k r in (join kx x l lt, gt)
    EQ -> (l,r)

  splitLookup :: (Num s, Ord s, Ord k) => k -> Dict' s k a -> (Maybe a, Dict' s k a, Dict' s k a)
  splitLookup k Tip = (Nothing, Tip, Tip)
  splitLookup k (Bin sx kx x l r) = case compare k kx of
    LT -> let (z, lt, gt) = splitLookup k l in (z, lt, join kx x gt r)
    GT -> let (z, lt, gt) = splitLookup k r in (z, join kx x l lt, gt)
    EQ -> (Just x, l, r)

  unions :: (Num s, Ord s, Ord k) => SizeProxy s -> [Dict' s k a] -> Dict' s k a
  unions p ts = foldl union (empty p) ts

  union :: (Num s, Ord s, Ord k) => Dict' s k a -> Dict' s k a -> Dict' s k a
  union t1 t2 =
    case t1 of
      Tip -> t2
      Bin _ _ _ _ _ -> case t2 of
        Tip -> t1
        Bin _ _ _ _ _
          | size t1 >= size t2 -> hedgeUnionL (const LT) (const GT) t1 t2
          | otherwise                  -> hedgeUnionR (const LT) (const GT) t2 t1

  hedgeUnionL :: (Num s, Ord s, Ord k) => (k -> Ordering) -> (k -> Ordering) -> Dict' s k a -> Dict' s k a -> Dict' s k a
  hedgeUnionL cmplo cmphi t1 t2 =
    case t2 of
      Tip -> t1
      Bin _ kx x l r -> case t1 of
        Tip -> join kx x (filterGt cmplo l) (filterLt cmphi r)
        Bin _ kx x l r ->
          let cmpkx k = compare kx k
          in join kx x (hedgeUnionL cmplo cmpkx l (trim cmplo cmpkx t2))
               (hedgeUnionL cmpkx cmphi r (trim cmpkx cmphi t2))

  hedgeUnionR :: (Num s, Ord s, Ord k) => (k -> Ordering) -> (k -> Ordering) -> Dict' s k a -> Dict' s k a -> Dict' s k a
  hedgeUnionR cmplo cmphi t1 t2 =
    case t2 of
      Tip -> t1
      Bin _ kx x l r ->
        case t1 of
          Tip -> join kx x (filterGt cmplo l) (filterLt cmphi r)
          Bin _ kx x l r ->
            let cmpkx k     = compare kx k
                lt          = trim cmplo cmpkx t2
                (found,gt)  = trimLookupLo kx cmphi t2
                newx        = case found of
                  Nothing -> x
                  Just y  -> y
            in join kx newx (hedgeUnionR cmplo cmpkx l lt)
                 (hedgeUnionR cmpkx cmphi r gt)

  unionWith :: (Num s, Ord s, Ord k) => (a -> a -> a) -> Dict' s k a -> Dict' s k a -> Dict' s k a
  unionWith f m1 m2 = unionWithKey (\k x y -> f x y) m1 m2

  unionWithKey :: (Num s, Ord s, Ord k) => (k -> a -> a -> a) -> Dict' s k a -> Dict' s k a -> Dict' s k a
  unionWithKey f t1 t2 = case t1 of
    Tip -> t2
    Bin _ _ _ _ _ -> case t2 of
      Tip -> t1
      Bin _ _ _ _ _
        | size t1 >= size t2 -> hedgeUnionWithKey f (const LT) (const GT) t1 t2
        | otherwise                  -> hedgeUnionWithKey flipf (const LT) (const GT) t2 t1
        where flipf k x y   = f k y x

  hedgeUnionWithKey :: (Num s, Ord s, Ord k) => (k -> a -> a -> a) -> (k -> Ordering) -> (k -> Ordering) -> Dict' s k a -> Dict' s k a -> Dict' s k a
  hedgeUnionWithKey f cmplo cmphi t1 t2 = case t2 of
    Tip -> t1
    Bin _ kx x l r -> case t1 of
      Tip -> join kx x (filterGt cmplo l) (filterLt cmphi r)
      Bin _ kx x l r ->
        let cmpkx k     = compare kx k
            lt          = trim cmplo cmpkx t2
            (found,gt)  = trimLookupLo kx cmphi t2
            newx        = case found of
              Nothing -> x
              Just y  -> f kx x y
        in join kx newx (hedgeUnionWithKey f cmplo cmpkx l lt)
             (hedgeUnionWithKey f cmpkx cmphi r gt)

  difference :: (Num s, Ord s, Ord k) => Dict' s k a -> Dict' s k a -> Dict' s k a
  difference t1 t2 = case t1 of
    Tip -> Tip
    Bin _ _ _ _ _ -> case t2 of
      Tip -> t1
      Bin _ _ _ _ _ -> hedgeDiff (const LT) (const GT) t1 t2

  hedgeDiff :: (Num s, Ord s, Ord k) => (k -> Ordering) -> (k -> Ordering) -> Dict' s k a -> Dict' s k a -> Dict' s k a
  hedgeDiff cmplo cmphi t1 t2 = case t1 of
    Tip -> Tip
    Bin _ kx x l r -> case t2 of
      Tip -> join kx x (filterGt cmplo l) (filterLt cmphi r)
      Bin _ kx x l r ->
        let cmpkx k = compare kx k
        in merge (hedgeDiff cmplo cmpkx (trim cmplo cmpkx t1) l)
             (hedgeDiff cmpkx cmphi (trim cmpkx cmphi t1) r)

  intersection :: (Num s, Ord s, Ord k) => Dict' s k a -> Dict' s k a -> Dict' s k a
  intersection m1 m2 = intersectionWithKey (\k x y -> x) m1 m2

  intersectionWith :: (Num s, Ord s, Ord k) => (a -> a -> a) -> Dict' s k a -> Dict' s k a -> Dict' s k a
  intersectionWith f m1 m2 = intersectionWithKey (\k x y -> f x y) m1 m2

  intersectionWithKey :: (Num s, Ord s, Ord k) => (k -> a -> a -> a) -> Dict' s k a -> Dict' s k a -> Dict' s k a
  intersectionWithKey f t1 t2 = case t1 of
    Tip -> Tip
    Bin _ _ _ _ _ -> case t2 of
      Tip -> Tip
      Bin _ _ _ _ _
        | size t1 >= size t2 -> intersectWithKey f t1 t2
        | otherwise          -> intersectWithKey flipf t2 t1
        where flipf k x y = f k y x

  intersectWithKey :: (Num s, Ord s, Ord k) => (k -> a -> a -> a) -> Dict' s k a -> Dict' s k a -> Dict' s k a
  intersectWithKey f t1 t2 = case t1 of
    Tip -> Tip
    Bin _ _ _ _ _ -> case t2 of
      Tip -> Tip
      Bin _ kx x l r ->
        let (found, lt, gt) = splitLookup kx t1
            tl              = intersectWithKey f lt l
            tr              = intersectWithKey f gt r
        in case found of
             Nothing -> merge tl tr
             Just y  -> join kx (f kx y x) tl tr

  fold :: (a -> b -> b) -> b -> Dict' s k a -> b
  fold f z m = foldWithKey (\k x z -> f x z) z m

  foldWithKey :: (k -> a -> b -> b) -> b -> Dict' s k a -> b
  foldWithKey f z t = foldR f z t

  foldR :: (k -> a -> b -> b) -> b -> Dict' s k a -> b
  foldR f z Tip              = z
  foldR f z (Bin _ kx x l r) = foldR f (f kx x (foldR f z r)) l

  foldL :: (b -> k -> a -> b) -> b -> Dict' s k a -> b
  foldL f z Tip              = z
  foldL f z (Bin _ kx x l r) = foldL f (f (foldL f z l) kx x) r

  assocs :: Dict' s k a -> [(k, a)]
  assocs m = toList m

  elems :: Dict' s k a -> [a]
  elems m = map snd (assocs m)

  keys  :: Dict' s k a -> [k]
  keys m = map fst (assocs m)

  fromList :: (Num s, Ord s, Ord k) => SizeProxy s -> [(k, a)] -> Dict' s k a
  fromList p xs = foldl ins (empty p) xs where
    ins :: (Num s, Ord s, Ord k) => Dict' s k a -> (k,a) -> Dict' s k a
    ins t (k,x) = insert k x t

  fromListWith :: (Num s, Ord s, Ord k) => SizeProxy s -> (a -> a -> a) -> [(k, a)] -> Dict' s k a
  fromListWith p f xs = fromListWithKey p (\k x y -> f x y) xs

  fromListWithKey :: (Num s, Ord s, Ord k) => SizeProxy s -> (k -> a -> a -> a) -> [(k, a)] -> Dict' s k a
  fromListWithKey p f xs = foldl (ins f) (empty p) xs where
    ins :: (Num s, Ord s, Ord k) => (k -> a -> a -> a) -> Dict' s k a -> (k, a) -> Dict' s k a
    ins f t (k,x) = insertWithKey f k x t

  toList :: Dict' s k a -> [(k,a)]
  toList t = toAscList t

  toAscList :: Dict' s k a -> [(k,a)]
  toAscList t = foldR (\k x xs -> (k, x) : xs) [] t

  toDescList :: Dict' s k a -> [(k, a)]
  toDescList t = foldL (\xs k x -> (k, x) : xs) [] t

  dictMap :: (a -> b) -> Dict' s k a -> Dict' s k b
  dictMap f m = mapWithKey (\k x -> f x) m

  mapWithKey :: (k -> a -> b) -> Dict' s k a -> Dict' s k b
  mapWithKey f Tip = Tip
  mapWithKey f (Bin sx kx x l r) = Bin sx kx (f kx x) (mapWithKey f l) (mapWithKey f r)

  mapAccum :: (a -> b -> (a,c)) -> a -> Dict' s k b -> (a, Dict' s k c)
  mapAccum f a m = mapAccumWithKey (\a k x -> f a x) a m

  mapAccumWithKey :: (a -> k -> b -> (a,c)) -> a -> Dict' s k b -> (a, Dict' s k c)
  mapAccumWithKey f a t = mapAccumL f a t

  mapAccumL :: (a -> k -> b -> (a,c)) -> a -> Dict' s k b -> (a, Dict' s k c)
  mapAccumL f a t = case t of
    Tip -> (a, Tip)
    Bin sx kx x l r ->
      let (a1, l') = mapAccumL f a l
          (a2, x') = f a1 kx x
          (a3, r') = mapAccumL f a2 r
      in (a3, Bin sx kx x' l' r')

  mapAccumR :: (a -> k -> b -> (a, c)) -> a -> Dict' s k b -> (a, Dict' s k c)
  mapAccumR f a t = case t of
    Tip -> (a, Tip)
    Bin sx kx x l r ->
      let (a1, r') = mapAccumR f a r
          (a2, x') = f a1 kx x
          (a3, l') = mapAccumR f a2 l
      in (a3, Bin sx kx x' l' r')

  dictFilter :: (Num s, Ord s, Ord k) => (a -> Bool) -> Dict' s k a -> Dict' s k a
  dictFilter p m = filterWithKey (\k x -> p x) m

  filterWithKey :: (Num s, Ord s, Ord k) => (k -> a -> Bool) -> Dict' s k a -> Dict' s k a
  filterWithKey p Tip = Tip
  filterWithKey p (Bin _ kx x l r) =
    if p kx x
    then join kx x (filterWithKey p l) (filterWithKey p r)
    else merge (filterWithKey p l) (filterWithKey p r)

  partition :: (Num s, Ord s, Ord k) => (a -> Bool) -> Dict' s k a -> (Dict' s k a, Dict' s k a)
  partition p m = partitionWithKey (\k x -> p x) m

  partitionWithKey :: (Num s, Ord s, Ord k) => (k -> a -> Bool) -> Dict' s k a -> (Dict' s k a, Dict' s k a)
  partitionWithKey p Tip = (Tip, Tip)
  partitionWithKey p (Bin _ kx x l r) =
    let (l1, l2) = partitionWithKey p l
        (r1, r2) = partitionWithKey p r
    in if p kx x
       then (join kx x l1 r1, merge l2 r2)
       else (merge l1 r1, join kx x l2 r2)

 |])

type Dict k a = Dict' Nat k a
