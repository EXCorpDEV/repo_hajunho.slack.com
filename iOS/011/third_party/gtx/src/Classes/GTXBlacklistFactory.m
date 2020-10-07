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

#import "GTXBlacklistFactory.h"

#import "GTXBlacklistBlock.h"

NS_ASSUME_NONNULL_BEGIN

@implementation GTXBlacklistFactory

+ (id<GTXBlacklisting>)blacklistWithClassName:(NSString *)elementClassName {
  Class classObject = NSClassFromString(elementClassName);
  NSAssert(classObject, @"Class named %@ does not exist!", elementClassName);
  GTXIgnoreElementMatcher matcher = ^BOOL(id element, NSString *checkName) {
    return [element isKindOfClass:classObject];
  };
  return [GTXBlacklistBlock blacklistWithBlock:matcher];
}

+ (id<GTXBlacklisting>)blacklistWithClassName:(NSString *)elementClassName
                                    checkName:(NSString *)skipCheckName {
  NSParameterAssert(elementClassName);
  Class classObject = NSClassFromString(elementClassName);
  NSAssert(classObject, @"Class named %@ does not exist!", elementClassName);
  GTXIgnoreElementMatcher matcher = ^BOOL(id element, NSString *checkName) {
    return [element isKindOfClass:classObject] && [checkName isEqualToString:skipCheckName];
  };
  return [GTXBlacklistBlock blacklistWithBlock:matcher];
}

+ (id<GTXBlacklisting>)blacklistWithAccessibilityIdentifier:(NSString *)accessibilityId
                                                  checkName:(NSString *)skipCheckName {
  NSParameterAssert(accessibilityId);
  NSParameterAssert(skipCheckName);
  GTXIgnoreElementMatcher matcher = ^BOOL(id element, NSString *checkName) {
    return [[element accessibilityIdentifier] isEqualToString:accessibilityId] &&
        [checkName isEqualToString:skipCheckName];
  };
  return [GTXBlacklistBlock blacklistWithBlock:matcher];
}

@end

NS_ASSUME_NONNULL_END
