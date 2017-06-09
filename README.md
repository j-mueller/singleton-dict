singleton-dict
==============

This package provides a typelevel balanced search tree based on an ancient version of Data.Map, 
originating from the uulib package. I used this older version as it uses relatively "simple"
Haskell, and thus is singletonized relatively straightforwardly.

Example
-------

````haskell
{-# LANGUAGE TypeOperators, PolyKinds, DataKinds, KindSignatures, ConstraintKinds #-}
{-# LANGUAGE TypeFamilies, UndecidableInstances, TemplateHaskell #-}
{-# LANGUAGE FlexibleInstances, GADTs, FlexibleContexts, ScopedTypeVariables, TypeInType #-}
{-# LANGUAGE ScopedTypeVariables, BangPatterns #-}

import Data.Promotion.Prelude.Maybe
import Data.Singletons
import Data.Singletons.TypeRepStar
import Data.Typeable
import Data.Type.Dict

type T0 = Empty NatProxy
type T1 = Insert "foo" Int T0
type T2 = Insert "bar" Bool T1
type T3 = Insert "baz" (Int -> Int) T2
type T4 = FromJust (T3 :!? "bar")

x :: T4
x = True

y :: Dict' Integer String TypeRep
y = fromSing (sing :: Sing T3)
````