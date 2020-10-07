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

#import "GTXTestBaseTest.h"

#import "GTXTestViewController.h"

static NSInteger gFailureCount = 0;
id<GTXChecking> checkFailsIfFailingClass;
id<GTXChecking> alwaysFail;
id<GTXChecking> alwaysPass;

@implementation GTXTestBaseTest

+ (void)setUp {
  [super setUp];

  checkFailsIfFailingClass = [GTXiLib checkWithName:@"checkFailsIfFailingClass"
                                              block:^BOOL(id element, GTXErrorRefType errorOrNil) {
    return ![element isKindOfClass:[GTXTestFailingClass class]];
  }];

  alwaysFail = [GTXiLib checkWithName:@"AlwaysFail"
                                block:^BOOL(id element, GTXErrorRefType errorOrNil) {
    return NO;
  }];
  alwaysPass = [GTXiLib checkWithName:@"AlwaysPass"
                                block:^BOOL(id element, GTXErrorRefType errorOrNil) {
    return YES;
  }];

  // Wait for view controller to be launched.
  NSTimeInterval start = CACurrentMediaTime();
  while (1) {
    NSAssert(CACurrentMediaTime() - start < 4.0, @"GTXTestViewController was not launched!");
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.1, true);
    if ([UIApplication sharedApplication].keyWindow.rootViewController.class ==
        [GTXTestViewController class]) {
      break;
    }
  }

  gFailureCount = 0;
  [GTXiLib setFailureHandler:^(NSError * error) {
    gFailureCount += 1;
  }];
}

- (void)assertFailureCount:(NSInteger)count {
  XCTAssertGreaterThan(count, 0, @"For 0 failures use -assertNoFailure");
  XCTAssertEqual(gFailureCount, count);
}

- (void)assertNoFailure {
  XCTAssertEqual(gFailureCount, 0);
}

- (void)assertAndClearSingleFailure {
  XCTAssertEqual(gFailureCount, 1);
  gFailureCount = 0;
}

@end

@implementation GTXTestPassingClass

- (void)drawRect:(CGRect)rect {
  [[UIColor greenColor] set];
  CGContextFillRect(UIGraphicsGetCurrentContext(), self.bounds);
}

- (BOOL)isAccessibilityElement {
  return YES;
}

@end

@implementation GTXTestFailingClass

- (void)drawRect:(CGRect)rect {
  [[UIColor redColor] set];
  CGContextFillRect(UIGraphicsGetCurrentContext(), self.bounds);
}

- (BOOL)isAccessibilityElement {
  return YES;
}

@end
