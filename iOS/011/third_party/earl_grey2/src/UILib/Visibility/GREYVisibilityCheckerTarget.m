//
// Copyright 2019 Google Inc.
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

#import "GREYVisibilityCheckerTarget.h"

#import "NSObject+GREYCommon.h"
#import "UIView+GREYCommon.h"
#import "CGGeometry+GREYUI.h"
#import "GREYScreenshotter.h"
#import "GREYVisibilityChecker.h"

/**
 * Minimum alpha value for a view to be considered obscuring. If an element is covered by a view
 * that has an alpha close to 1, we assume that the user does not intend to match or interact with
 * the element. For instance, if a view with alpha 0.95 covers @c _target, @c _target is deemed not
 * visible.
 */
const double kMinimumAlphaToConsiderAsObscuring = 0.95f;

@implementation GREYVisibilityCheckerTarget {
  /**
   * Internal target element. Target is an accessibility element that could be either a UIView or a
   * non-UIView instance.
   */
  id _target;
  /**
   * View that contains @c _target.
   */
  UIView *_targetContainerView;
  /**
   * Boolean whether or not @c _target is a UIView.
   */
  BOOL _isView;
  /**
   * Visible frame of the @c _target in screen coordinate.
   */
  CGRect _targetRect;
  /**
   * CGRect representation of @c _bitVector.
   */
  CGRect _bitVectorRect;
  /**
   * Binary bitmap representing the visible portion of @c _target in pixel coordinate. Each pixel
   * contains either 0 or 1. 0 indicating the pixel is visible, 1 if not. All rects must be
   * converted to pixel and aligned before interacting with @c _bitVector.
   */
  CFMutableBitVectorRef _bitVector;
  /**
   * Intersections between the views and the @c _target in pixel coordinate. These intersections
   * will be subtracted from the @c _bitVector when the traversal is finished.
   */
  NSMutableArray<NSValue *> *_intersections;
  /**
   * Temp variable that checks if an element meets the condition to fall back to the through
   * visibility checker.
   */
  BOOL _currentElementMeetsFallbackCondition;
  /**
   * Whether or not the @c _target should consider interactability into account when being obscured
   * by elements. If it should be interactable, @c _bitVector would only show portion of the @c
   * _target not only visible, but also interactable by the user.
   */
  BOOL _interactability;
}

- (instancetype)initWithObject:(GREYTraversalObject *)object interactability:(BOOL)interactability {
  id target = object.element;
  UIView *containerView = [target grey_viewContainingSelf];
  BOOL isView = [target isKindOfClass:[UIView class]];
  CGRect targetRect = VisibleRectOnScreen(object);
  CGRect bitVectorRect = CGRectPointToPixelAligned(targetRect);
  if (!IsElementVisible(object)) {
    // Check if target is visible or not.
    return nil;
  } else if (CGRectIsEmpty(targetRect)) {
    // Check if target is visible on screen.
    return nil;
  } else if (CGRectGetWidth(bitVectorRect) < 1 || CGRectGetHeight(bitVectorRect) < 1) {
    // Check for views that are smaller than a point in the screen.
    return nil;
  }
  self = [super init];
  if (self) {
    _target = target;
    _targetContainerView = containerView;
    _isView = isView;
    _intersections = [[NSMutableArray alloc] init];
    _targetRect = targetRect;
    _bitVectorRect = bitVectorRect;
    CGFloat bitVectorSize = CGRectArea(_bitVectorRect);
    // bit vector is initialized with 0 by default.
    _bitVector = CFBitVectorCreateMutable(kCFAllocatorDefault, 0);
    CFBitVectorSetCount(_bitVector, (CFIndex)bitVectorSize);
    _interactability = interactability;
  }
  return self;
}

/**
 * To calculate the percentage visible from the target view, we maintain a bit matrix that keeps
 * track of the pixels that are obscured by other views, represented as 1. Once we gather all
 * intersections between the target and other views, we go through each intersections and set the
 * bits from the bit matrix that falls into that intersection rect. After we are done setting all
 * bits, we could calculate the visible percentage of the target as we can count how many pixels
 * are visible in the bit matrix (by counting 0's).
 *
 * @return percentage visible from the target element.
 */
- (CGFloat)percentageVisible {
  [self calculateBitsForIntersectionsInParallel];
  CGFloat bitVectorSize = CGRectArea(_bitVectorRect);
  NSInteger numberOfVisiblePixels =
      (NSInteger)CFBitVectorGetCountOfBit(_bitVector, CFRangeMake(0, (CFIndex)bitVectorSize), 0);
  CGRect frame = _isView ? [_target frame] : [_target accessibilityFrame];
  CGRect targetOriginalPixelRect = CGRectPointToPixelAligned(frame);
  return numberOfVisiblePixels / CGRectArea(targetOriginalPixelRect);
}

- (GREYVisibilityCheckerTargetObscureResult)overlapResultWithObject:(GREYTraversalObject *)object
                                                           fallback:(BOOL *)fallback {
  if ([self shouldSkipObject:object]) {
    return GREYVisibilityCheckerTargetObscureResultNone;
  }
  CGRect viewRect = VisibleRectOnScreen(object);
  if (CGRectIsNull(viewRect)) {
    return GREYVisibilityCheckerTargetObscureResultNone;
  }
  CGRect intersection = CGRectIntersectionStrict(_targetRect, viewRect);
  // Check if intersection area is zero in case the _targetRect and viewRect overlap in the border.
  if (CGRectIsNull(intersection) || CGRectArea(intersection) == 0) {
    return GREYVisibilityCheckerTargetObscureResultNone;
  }
  // Since the target intersects with the current element, set the fallback flag.
  *fallback = _currentElementMeetsFallbackCondition;
  NSValue *rectValue = [NSValue valueWithCGRect:intersection];
  [_intersections addObject:rectValue];
  // If intersection and _targetRect is the same, it means _targetRect is completely obscured, and
  // the traversal can be stopped prematurely.
  return CGRectEqualToRect(intersection, _targetRect)
             ? GREYVisibilityCheckerTargetObscureResultFull
             : GREYVisibilityCheckerTargetObscureResultPartial;
}

/**
 * There are few points within the @c _target we want to check for interaction points.
 * (1) accessibilityActivationPoint
 * (2) Center points of each quadrant of @c _targetRect.
 * (3) Center point of the largest interactable rect. You want to check this last since it's most
 *     expensive.
 * The points are checked in numerical order.
 *
 * @return Point in @c _target that is interactable by the user. If none of the above points are
 *         interactable, @c GREYCGPointNull is returned.
 */
- (CGPoint)interactionPoint {
  // Check if the _target is at least @c kMinimumPointsVisibleForInteraction large.
  size_t widthInPoints = (size_t)CGRectGetWidth(_targetRect);
  size_t heightInPoints = (size_t)CGRectGetHeight(_targetRect);
  if ((widthInPoints * heightInPoints) < kMinimumPointsVisibleForInteraction) {
    return GREYCGPointNull;
  }

  [self calculateBitsForIntersectionsInParallel];

  // Check for accessibilityActivationPoint.
  CGPoint accessibilityActivationPoint = [_target accessibilityActivationPoint];
  if ([self isInteractableAtPointInScreenCoordinate:accessibilityActivationPoint]) {
    return [self localInteractionPointFromScreenCoordinate:accessibilityActivationPoint];
  }

  // Check for center points of each quadrants of targetRect, and return whichever is interactable.
  for (NSValue *pointInValue in [self centersOfTargetRectQuadrants]) {
    CGPoint interactionPoint = [pointInValue CGPointValue];
    if ([self isInteractableAtPointInScreenCoordinate:interactionPoint]) {
      return [self localInteractionPointFromScreenCoordinate:interactionPoint];
    }
  }

  // If all previous interaction points are not interactable, look for the center of largest
  // interactable area of the target. This should always return a valid interactable point unless
  // the largestRect is less than the kMinimumPointsVisibleForInteraction.
  CGRect largestRect = [self largestInteractableRect];
  if (CGRectArea(largestRect) < kMinimumPointsVisibleForInteraction) {
    return GREYCGPointNull;
  }
  CGPoint center = CGPointMake(CGRectGetMidX(largestRect), CGRectGetMidY(largestRect));
  CGPoint centerInScreenCoordinate =
      CGPointMake(center.x + CGRectGetMinX(_targetRect), center.y + CGRectGetMinY(_targetRect));
  // We assume that centerInScreenCoordinate is already interactable.
  return [self localInteractionPointFromScreenCoordinate:centerInScreenCoordinate];
}

#pragma mark - Private

/**
 * Dispatches each intersections to set bits in each intersection rect from @c _bitVector. Only
 * call this method when there's no more intersection that is obscuring @c _targetRect.
 */
- (void)calculateBitsForIntersectionsInParallel {
  dispatch_apply(_intersections.count, DISPATCH_APPLY_AUTO, ^(size_t idx) {
    CGRect intersection = [self->_intersections[idx] CGRectValue];
    [self setBitsInRect:intersection];
  });
}

/**
 * Sets all bits inside @c rect from @c _bitVector. The @c rect is converted to pixels as per
 * screen scale before setting the bits. This is performed in parallel across multiple threads.
 *
 * @param rect The frame of the pixels to set the bits in @c _bitVector. Must be in points.
 */
- (void)setBitsInRect:(CGRect)rect {
  rect = CGRectPointToPixelAligned(rect);
  // _targetRect is indirectly translated to (0,0) since bitVector starts from (0, 0). Therefore,
  // the rect needs to be translated as much as _bitVectorRect did towards the origin.
  CGRect translatedRect = CGRectMake(CGRectGetMinX(rect) - CGRectGetMinX(_bitVectorRect),
                                     CGRectGetMinY(rect) - CGRectGetMinY(_bitVectorRect),
                                     CGRectGetWidth(rect), CGRectGetHeight(rect));
  NSInteger bitVectorRectWidth = (NSInteger)CGRectGetWidth(_bitVectorRect);
  NSInteger bitVectorRectHeight = (NSInteger)CGRectGetHeight(_bitVectorRect);
  NSInteger lowerX = (NSInteger)CGRectGetMinX(translatedRect);
  NSInteger lowerY = (NSInteger)CGRectGetMinY(translatedRect);
  NSInteger upperY = (NSInteger)CGRectGetMaxY(translatedRect);
  NSInteger width = (NSInteger)CGRectGetWidth(translatedRect);
  for (NSInteger y = lowerY; y < upperY; y++) {
    // Pixel aligned rect
    if (y < bitVectorRectHeight) {
      CFIndex start = y * bitVectorRectWidth + lowerX;
      CFRange range = CFRangeMake(start, MIN(width, bitVectorRectWidth - lowerX));
      CFBitVectorSetBits(_bitVector, range, 1);
    }
  }
}

/**
 * Evaluates whether or not we should skip @c object from calculation because it cannot obscure the
 * target due to its visual property. A target cannot be obscured if the other view drawn on top of
 * the target has the following conditions:
 * (1) Accessibility element that is not a UIView. It cannot obscure the @c _target because it's
 *     not a visual element.
 * (2) Its backgroundColor has an alpha less than 0.95.
 * (3) Its alpha is less than 0.95.
 * (4) It is hidden or any of its ancestor is hidden.
 *
 * @param object The object to evaluate if it can potentially obscure the @c _target. It could be
 *               either a view or an accessibility element (non view).
 *
 * @return A BOOL whether or not @c object can potentially obscure the @c _target.
 */
- (BOOL)shouldSkipObject:(GREYTraversalObject *)object {
  id element = object.element;
  BOOL elementIsView = [element isKindOfClass:[UIView class]];
  _currentElementMeetsFallbackCondition = NO;
  if (!_isView && [_target isAccessibilityElement]) {
    // If the target element is an accessibility element, it cannot be obscured by
    // any of its accessibility container's subviews.
    UIView *view = elementIsView ? element : [element grey_viewContainingSelf];
    if ([_targetContainerView grey_isAncestorOfView:view]) {
      return YES;
    }
  }
  if (!elementIsView) {
    // If element is not a UIView, it should not obscure the target.
    return YES;
  }

  // element is a view.
  UIView *view = (UIView *)element;
  if (!IsElementVisible(object)) {
    // Check if view is hidden or has alpha less than 0.01.
    return YES;
  } else if ([NSStringFromClass([view class]) isEqualToString:@"_UIVisualEffectBackdropView"]) {
    // _UIVisualEffectBackdropView is an iOS internal view that is used for blurred view. This
    // includes, but not limited to, system navigation bar, system keyboard, etc. Elements behind
    // this view should not be considered visible. Custom drawn view as this does not have any
    // distinguishable traits in the UIView property that could rule it as an "obscuring view" from
    // the current algorithm.
    // TODO(b/146083877): Add support for custom drawn views.
    return NO;
  } else if (MeetsFallbackCondition(view)) {
    // This doesn't necessarily mean you should perform a fallback. It doesn't have to fallback if
    // view doesn't intersect with the target view.
    _currentElementMeetsFallbackCondition = YES;
    return NO;
  } else if (IsBackgroundColorTranslucent(view.backgroundColor)) {
    return YES;
  } else if (view.alpha < kMinimumAlphaToConsiderAsObscuring) {
    return YES;
  } else {
    return NO;
  }
}

/**
 * Provides an array of possible interaction points obtained from the centers of each quadrant of
 * @c _targetRect. This is not necessarily the same as the centers of each quadrant of @c _target's
 * bounds because, as noted above, @c _targetRect represents the frame that is visible on
 * screen. The following points are added to the array:
 *
 * (1) center of the @c _targetRect's first quadrant.
 * (2) center of the @c _targetRect's second quadrant.
 * (3) center of the @c _targetRect's third quadrant.
 * (4) center of the @c _targetRect's forth quadrant.
 *
 * @return An array of center points in each quadrant of the @c _targetRect in screen coordinate.
 */
- (NSArray<NSValue *> *)centersOfTargetRectQuadrants {
  CGFloat minX = CGRectGetMinX(_targetRect);
  CGFloat minY = CGRectGetMinY(_targetRect);
  CGFloat maxX = CGRectGetMaxX(_targetRect);
  CGFloat maxY = CGRectGetMaxY(_targetRect);
  CGFloat numeratorConstant = 3.0f;
  CGFloat denominatorConstant = 4.0f;
  CGFloat xLeftQuadrant = (numeratorConstant * minX + maxX) / denominatorConstant;
  CGFloat xRightQuardrant = (minX + numeratorConstant * maxX) / denominatorConstant;
  CGFloat yTopQuardrant = (numeratorConstant * minY + maxY) / denominatorConstant;
  CGFloat yBottomQuardrant = (minY + numeratorConstant * maxY) / denominatorConstant;
  CGPoint centerOfFirstQuadrant = CGPointMake(xRightQuardrant, yTopQuardrant);
  CGPoint centerOfSecondQuadrant = CGPointMake(xLeftQuadrant, yTopQuardrant);
  CGPoint centerOfThirdQuadrant = CGPointMake(xLeftQuadrant, yBottomQuardrant);
  CGPoint centerOfForthQuadrant = CGPointMake(xRightQuardrant, yBottomQuardrant);
  NSArray<NSValue *> *possibleInteractionPoints = @[
    [NSValue valueWithCGPoint:centerOfFirstQuadrant],
    [NSValue valueWithCGPoint:centerOfSecondQuadrant],
    [NSValue valueWithCGPoint:centerOfThirdQuadrant],
    [NSValue valueWithCGPoint:centerOfForthQuadrant]
  ];
  return possibleInteractionPoints;
}

/**
 * Converts an interaction point in screen coordinate to local coordinate in @c _target's bounds.
 *
 * @return CGPoint specifying the interaction point in @c _target's bounds.
 */
- (CGPoint)localInteractionPointFromScreenCoordinate:(CGPoint)pointInScreenCoordinate {
  CGRect accessibilityFrame = [_target accessibilityFrame];
  CGPoint localInteractionPoint;
  if (_isView) {
    UIView *view = (UIView *)_target;
    pointInScreenCoordinate = [view.window convertPoint:pointInScreenCoordinate fromWindow:nil];
    localInteractionPoint = [view convertPoint:pointInScreenCoordinate fromView:nil];
  } else {
    // target is not a view
    localInteractionPoint =
        CGPointMake(pointInScreenCoordinate.x - CGRectGetMinX(accessibilityFrame),
                    pointInScreenCoordinate.y - CGRectGetMinY(accessibilityFrame));
  }
  return localInteractionPoint;
}

/**
 * Intersects with the screen bounds and element's bounding area to cut off any portion of the
 * element that is not visible on screen.
 *
 * @param object Traversing object to check for the visible rect in screen.
 *
 * @return CGRect specifying the frame of the element that is visible on screen in screen
 *         coordinate. @c CGRectNull if it is not visible at all.
 */
static CGRect VisibleRectOnScreen(GREYTraversalObject *object) {
  id element = object.element;
  CGRect boundingRect = object.properties.boundingRect;
  UIView *container = [element grey_viewContainingSelf];
  CGRect containerRect = ConvertToScreenCoordinate(container);
  // If element has a mask view, use that mask view instead.
  if ([element isKindOfClass:[UIView class]]) {
    UIView *maskView = [element maskView];
    if (maskView) {
      element = maskView;
    }
  }

  CGRect elementRect = ConvertToScreenCoordinate(element);
  elementRect = CGRectIntersectionStrict(elementRect, [UIScreen mainScreen].bounds);
  if (!CGRectIsNull(boundingRect)) {
    CGRect boundingRectScreenCoord = [container convertRect:boundingRect toView:nil];
    elementRect = CGRectIntersectionStrict(elementRect, boundingRectScreenCoord);
  }
  // If the element is a subview of a UIScrollView, it should be bounded by the UIScrollView's
  // bound.
  if ([container isKindOfClass:[UIScrollView class]] && [container clipsToBounds]) {
    elementRect = CGRectIntersectionStrict(elementRect, containerRect);
  }
  return elementRect;
}

/**
 * Converts @c element's frame to screen coordinate. This is needed before intersecting views with
 * the @c _target.
 *
 * @param element The element whose frame will be converted.
 *
 * @return @c element's frame converted to screen coordinate.
 */
static CGRect ConvertToScreenCoordinate(id element) {
  // Although UIView's accessibilityFrame, by default, is in screen-coordinate, we should be using
  // -[UIView convertRect:toView:] as the accessibilityFrame can be changed to any value by the
  // user. So, the frame could be different from where the actual element is located.
  if ([element isKindOfClass:[UIView class]]) {
    UIView *container = [element grey_viewContainingSelf];
    if (container) {
      // TODO(b/160199305): For iOS 13+, when a UITableViewCell is in delete mode, the element's y
      // position is shifted by 44 than the actual position of the element, resulting in a wrong
      // screen coordinate. Since element's accessibilityFrame has the correct frame, this should be
      // a workaround until Apple fixes this bug.
      if ([container isKindOfClass:NSClassFromString(@"_UITableViewCellSwipeContainerView")]) {
        return [element accessibilityFrame];
      }
      return [container convertRect:[element frame] toView:nil];
    } else {
      // For top-level UIWindows
      return [element frame];
    }
  } else {
    return [element accessibilityFrame];
  }
}

/**
 * @return A @c BOOL if the traversal object is visible or not. This is different from
 *         checking @c hidden and @c alpha property from UIView as @c properties are derived from
 *         the element's ancestors during traversal.
 */
static BOOL IsElementVisible(GREYTraversalObject *object) {
  return !(object.properties.hidden || object.properties.lowestAlpha < kGREYMinimumVisibleAlpha);
}

/**
 * If the following criteria is met, a fallback should be performed depending on the position of
 * the @c view as the quick visibility checker can no longer provide an accurate answer. If the
 * view does not intersect with the target, fallback does not have to be performed.
 *
 * Assuming @c view intersects with @c _target, perform fallback if:
 * (1) @c view is transformed: Because quick visibility checker performs a frame by frame
 *     comparison, it cannot accurately determine the visibility of the target under a transformed
 *     view.
 * (2) @c view contains @c CAShapeLayer: Since quick visibility checker doesn't support custom
 *     drawn views yet, it cannot accurately obtain the visibility of an element obscured by a
 *     custom drawn view with CAShapeLayer.
 *
 * @param view @c UIView that is drawn on top of target in the view hierarchy.
 *
 * @return Whether or not the @c view meets the fallback condition.
 */
static BOOL MeetsFallbackCondition(UIView *view) {
  if (!CGAffineTransformEqualToTransform(CGAffineTransformIdentity, [view transform])) {
    return YES;
  } else if (IsBackgroundColorTranslucent(view.backgroundColor)) {
    // TODO(b/146083877): Add support for custom drawn views. This check should be removed once
    // quick visibility checker supports custom drawn views.
    CALayer *backingLayer = view.layer;
    static Class shapeLayerClass = nil;
    if (!shapeLayerClass) {
      shapeLayerClass = [CAShapeLayer class];
    }
    if ([backingLayer isKindOfClass:shapeLayerClass]) {
      return YES;
    }
    for (CALayer *layer in backingLayer.sublayers) {
      if ([layer isKindOfClass:shapeLayerClass]) {
        return YES;
      }
    }
    static Class imageViewClass = nil;
    if (!imageViewClass) {
      imageViewClass = [UIImageView class];
    }
    if ([view isKindOfClass:imageViewClass] && ((UIImageView *)view).image) {
      return YES;
    }
  }
  return NO;
}

/**
 * @param backgroundColor Color to check for translucency.
 *
 * @return A @c BOOL indicating whether or not you can see through the @c backgroundColor.
 */
static BOOL IsBackgroundColorTranslucent(UIColor *backgroundColor) {
  if (!backgroundColor || [backgroundColor isEqual:UIColor.clearColor]) {
    return YES;
  } else {
    CGFloat white;
    CGFloat alpha;
    BOOL success = [backgroundColor getWhite:&white alpha:&alpha];
    return success && (alpha < kMinimumAlphaToConsiderAsObscuring);
  }
}

/**
 * @return A @c BOOL indicating whether or not the @c pointInScreenCoordinate is interactable by
 *         the user. @c pointInScreenCoordinate must be in screen coordinate system.
 */
- (BOOL)isInteractableAtPointInScreenCoordinate:(CGPoint)pointInScreenCoordinate {
  // Check if this point lies in the screen bounds.
  CGRect screenBounds = [UIScreen mainScreen].bounds;
  if (!CGRectContainsPoint(screenBounds, pointInScreenCoordinate)) {
    return NO;
  }
  // Convert the point in screen coordinate to the coordinate in _targetRect
  CGPoint pointInTargetRectCoordinate =
      CGPointMake(pointInScreenCoordinate.x - CGRectGetMinX(_targetRect),
                  pointInScreenCoordinate.y - CGRectGetMinY(_targetRect));

  // Out of range
  if (pointInTargetRectCoordinate.x < 0 ||
      pointInTargetRectCoordinate.x >= CGRectGetWidth(_targetRect)) {
    return NO;
  }
  if (pointInTargetRectCoordinate.y < 0 ||
      pointInTargetRectCoordinate.y >= CGRectGetHeight(_targetRect)) {
    return NO;
  }

  // Convert point to bitvector index
  CGPoint pixelCoordinate = CGPointToPixel(pointInTargetRectCoordinate);

  NSUInteger index =
      (NSUInteger)(pixelCoordinate.y * CGRectGetWidth(_bitVectorRect) + pixelCoordinate.x);
  return (BOOL)!CFBitVectorGetBitAtIndex(_bitVector, (CFIndex)index);
}

/**
 * @return The largest interactable area of @c _target in points.
 */
- (CGRect)largestInteractableRect {
  NSInteger width = (NSInteger)CGRectGetWidth(_bitVectorRect);
  NSInteger height = (NSInteger)CGRectGetHeight(_bitVectorRect);
  uint16_t *histograms = calloc((size_t)(width * height), sizeof(uint16_t));

  for (NSInteger y = 0; y < height; y++) {
    for (NSInteger x = 0; x < width; x++) {
      NSInteger currentPixelIndex = y * width + x;
      BOOL pixelIsVisible = !CFBitVectorGetBitAtIndex(_bitVector, (CFIndex)currentPixelIndex);
      // We don't care about the first byte because we are dealing with XRGB format.
      if (y == 0) {
        histograms[x] = pixelIsVisible ? 1 : 0;
      } else {
        histograms[y * width + x] = pixelIsVisible ? (histograms[(y - 1) * width + x] + 1) : 0;
      }
    }
  }

  CGRect largestRect = CGRectZero;
  for (NSInteger idx = 0; idx < height; idx++) {
    CGRect thisLargest = CGRectLargestRectInHistogram(&histograms[idx * width], (uint16_t)width);
    if (CGRectArea(thisLargest) > CGRectArea(largestRect)) {
      // Because our histograms point up, not down.
      thisLargest.origin.y = idx - thisLargest.size.height + 1;
      largestRect = thisLargest;
    }
  }

  free(histograms);
  return CGRectPixelToPoint(largestRect);
}

- (void)dealloc {
  if (_bitVector) {
    CFRelease(_bitVector);
    _bitVector = NULL;
  }
}

@end
