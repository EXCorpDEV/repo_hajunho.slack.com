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

#import "GREYTraversalBFS.h"

#import "GREYThrowDefines.h"
#import "GREYTraversalFunctions.h"
#import "GREYTraversalObject.h"

@implementation GREYTraversalBFS {
  /**
   * An array that contains the unrolled hierarchy.
   */
  NSMutableArray *_parsedHierarchy;

  /**
   * NSUInteger to keep track of the index that is being accessed in the @c _parsedHierarchy array.
   */
  NSUInteger _parsedHierarchyIndex;
}

- (instancetype)initWithElement:(id)element {
  self = [super init];
  if (self) {
    _parsedHierarchy = [[NSMutableArray alloc] init];
    _parsedHierarchyIndex = 0;

    GREYTraversalObject *object = [[GREYTraversalObject alloc] initWithElement:element level:0];
    [_parsedHierarchy addObject:object];
  }
  return self;
}

+ (instancetype)hierarchyForElementWithBFSTraversal:(id)element {
  GREYThrowOnNilParameter(element);
  // Create an instance of GREYTraversalBFS object.
  return [[GREYTraversalBFS alloc] initWithElement:element];
}

- (id)nextObject {
  GREYTraversalObject *nextObject = [self grey_nextObjectBFS];
  return nextObject.element;
}

- (void)enumerateUsingBlock:(void (^)(GREYTraversalObject *object, BOOL *stop))block {
  GREYThrowOnNilParameter(block);

  GREYTraversalObject *object;
  BOOL stop = NO;
  while (!stop && (object = [self grey_nextObjectBFS])) {
    block(object, &stop);
  }
}

/**
 * The method retrieves the next object in the hierarchy.
 *
 * @return Returns an instance of GREYTraversalObject.
 */
- (GREYTraversalObject *)grey_nextObjectBFS {
  // If we have explored all elements, then we return nil.
  if (_parsedHierarchyIndex >= [_parsedHierarchy count]) {
    return nil;
  }

  GREYTraversalObject *nextObject = [_parsedHierarchy objectAtIndex:_parsedHierarchyIndex];
  ++_parsedHierarchyIndex;

  // Ask GREYTraversal i.e. parent class for the immediate children of @c nextObject.
  NSArray<id> *children = GREYTraversalExploreImmediateChildren(nextObject.element, NO);
  for (id child in children) {
    GREYTraversalObject *object =
        [[GREYTraversalObject alloc] initWithElement:child level:nextObject.level + 1];
    [_parsedHierarchy addObject:object];
  }
  return nextObject;
}

@end
