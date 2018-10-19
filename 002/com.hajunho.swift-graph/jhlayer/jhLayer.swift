//
//  jhLayer.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 19..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

class jhLayer : CALayer {
    
    internal var mValuesOfDatas : Array<CGFloat> = Array()
    
    var axisDistance, mVerticalRatioToDraw_view, mMargin, mPanelWidth, mPanelHeight, mFixedPanelWidth, mFixedPanelHeight : CGFloat
    
    init(_ value: inout Array<CGFloat>, _ axisD: CGFloat, _ vRatio: CGFloat, _ margin: CGFloat, _ w: CGFloat, _ h: CGFloat, _ fw: CGFloat, _ fh: CGFloat, layer: Any) {
        
        self.mValuesOfDatas = value
        self.axisDistance = axisD
        self.mVerticalRatioToDraw_view = vRatio
        self.mMargin = margin
        self.mPanelWidth = w
        self.mPanelHeight = h
        self.mFixedPanelWidth = fw
        self.mFixedPanelHeight = fh
        
        super.init(layer: layer)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(in ctx: CGContext) {
        //        worldEllipse(context: mContext, 100, 100, 100, 100, 2, UIColor.blue.cgColor)
        var pointCloud = Array<CGPoint>()
        var fx, fy : CGFloat
        
        var x : Int = 0
        for y in mValuesOfDatas {
            //ref:drawLine(CGFloat(x)*axisDistance + mMargin, mMargin, CGFloat(x) * axisDistance + mMargin, 10000-mMargin)
            x += 1
            fx = CGFloat(x)*axisDistance
            fy = CGFloat(y)*mVerticalRatioToDraw_view + mMargin
            drawEllipse2(ctx, fx, fy, 2, 2, thickness: 2, UIColor.blue.cgColor)
            pointCloud.append(CGPoint.init(x: getX(fx+mMargin)!, y: getY(fy)!))
        }
        
        ctx.move(to: CGPoint.init(x: 0, y: 0))
        ctx.setStrokeColorSpace(CGColorSpaceCreateDeviceRGB())
        ctx.setStrokeColor(UIColor.blue.cgColor)
        ctx.setLineWidth(1.0)
        ctx.addLines(between: pointCloud)
        ctx.strokePath()
    }
    
    func drawEllipse2(_ ctx: CGContext, _ x : CGFloat, _ y : CGFloat, _ width : CGFloat, _ height : CGFloat, thickness : CGFloat, _ color : CGColor){
        //        worldEllipse(context: mContext, getX(x)!, getY(jhDraw.maxR - y)!, width, height, thickness, color)
        if GS.shared.logLevel.contains(.graph) {
            print("worldEllipse(context: mContext,", getX(x+mMargin)!, getY(jhDraw.maxR-y)!, width, height, thickness, color)
        }
        jhDraw.worldEllipse(context: ctx, getX(x+mMargin)!, getY(y)!, width, height, thickness, color)
    }
    
    func getX(_ x: CGFloat) -> CGFloat? {
        var retX : CGFloat? = nil
        retX = x * mPanelWidth / mFixedPanelWidth
        return retX
    }
    
    func getY(_ y: CGFloat) -> CGFloat? {
        var retY : CGFloat? = nil
        retY = y * mPanelHeight / mFixedPanelHeight
        return retY
    }
}
