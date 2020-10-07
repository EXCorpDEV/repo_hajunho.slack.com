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

#import "GTXTestStepperButton.h"

@implementation GTXTestStepperButton

- (void)touchesEnded:(NSSet<UITouch *> *)touches withEvent:(UIEvent *)event {
  [self increment];
}

- (void)increment {
  _stepCount += 1;
  [self setNeedsDisplay];
}

- (void)drawRect:(CGRect)rect {
  // Render the string: "<count> ↑" into the element bounds.
  CGContextRef context  = UIGraphicsGetCurrentContext();
  CGContextSetFillColorWithColor(context, [UIColor lightGrayColor].CGColor);
  CGContextFillRect(context, rect);
  CGContextSetStrokeColorWithColor(context, [UIColor redColor].CGColor);
  CGContextStrokeRect(context, rect);
  NSString *text = [NSString stringWithFormat:@"%u ↑", (unsigned int)_stepCount];
  [text drawInRect:rect withAttributes:@{ NSForegroundColorAttributeName: [UIColor redColor],
                                          NSFontAttributeName: [UIFont systemFontOfSize:35] }];
}

- (BOOL)accessibilityActivate {
  [self increment];
  return YES;
}

@end
