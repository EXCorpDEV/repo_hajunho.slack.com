//
// Copyright 2017 Google Inc.
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

#import "GREYActionBlock.h"

#import "GREYThrowDefines.h"
#import "GREYDefines.h"
#import "GREYMatcher.h"

@implementation GREYActionBlock {
  GREYPerformBlock _performBlock;
  /**
   * Identifier used for diagnostics.
   */
  NSString *_diagnosticsID;
}

+ (instancetype)actionWithName:(NSString *)name performBlock:(GREYPerformBlock)block {
  return [GREYActionBlock actionWithName:name constraints:nil performBlock:block];
}

+ (instancetype)actionWithName:(NSString *)name
                   constraints:(id<GREYMatcher>)constraints
                  performBlock:(GREYPerformBlock)block {
  return [[GREYActionBlock alloc] initWithName:name constraints:constraints performBlock:block];
}

- (instancetype)initWithName:(NSString *)name
                 constraints:(id<GREYMatcher>)constraints
                performBlock:(GREYPerformBlock)block {
  GREYThrowOnNilParameter(block);

  self = [super initWithName:name constraints:constraints];
  if (self) {
    _performBlock = [block copy];
  }
  return self;
}

#pragma mark - Private

+ (instancetype)actionWithName:(NSString *)name
                 diagnosticsID:(NSString *)diagnosticsID
                   constraints:(id<GREYMatcher>)constraints
                  performBlock:(GREYPerformBlock)block {
  return [[GREYActionBlock alloc] initWithName:name
                                 diagnosticsID:diagnosticsID
                                   constraints:constraints
                                  performBlock:block];
}

- (instancetype)initWithName:(NSString *)name
               diagnosticsID:(NSString *)diagnosticsID
                 constraints:(id<GREYMatcher>)constraints
                performBlock:(GREYPerformBlock)block {
  self = [self initWithName:name constraints:constraints performBlock:block];
  if (self) {
    _diagnosticsID = diagnosticsID;
  }
  return self;
}

#pragma mark - GREYAction

- (BOOL)perform:(id)element error:(__strong NSError **)errorOrNil {
  if (![self satisfiesConstraintsForElement:element error:errorOrNil]) {
    return NO;
  }
  // Perform actual action.
  return _performBlock(element, errorOrNil);
}

#pragma mark - GREYDiagnosable

- (NSString *)diagnosticsID {
  return _diagnosticsID;
}

@end
