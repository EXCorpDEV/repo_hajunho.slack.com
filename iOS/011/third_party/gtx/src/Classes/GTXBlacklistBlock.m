//
// Copyright 2018 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#import "GTXBlacklistBlock.h"

NS_ASSUME_NONNULL_BEGIN

@implementation GTXBlacklistBlock {
  GTXIgnoreElementMatcher _block;
}

+ (id<GTXBlacklisting>)blacklistWithBlock:(GTXIgnoreElementMatcher)block {
  return [[GTXBlacklistBlock alloc] initWithBlock:block];
}

- (instancetype)initWithBlock:(GTXIgnoreElementMatcher)block {
  NSParameterAssert(block);

  self = [super init];
  if (self) {
    _block = block;
  }
  return self;
}

#pragma mark - GTXBlacklisting

- (BOOL)shouldIgnoreElement:(id)element forCheckNamed:(NSString *)check {
  return _block(element, check);
}

@end

NS_ASSUME_NONNULL_END
